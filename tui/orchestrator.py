"""Autonomous experiment orchestrator.

Drives the experiment loop: LLM generates code → commit → train → evaluate →
keep/discard → repeat. Runs in a background thread and communicates with the
TUI via callbacks.

Resilience features:
- Resumes from interrupted runs (reads existing results.tsv)
- Exponential backoff on API errors (rate limits, 5xx, connection issues)
- Mid-experiment cleanup on startup (reverts orphaned commits)
- Heartbeat file for external monitoring
"""

import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from tui.git_manager import GitManager
from tui.hardware import get_hardware_summary
from tui.llm_backend import LLMBackend, get_llm_backend
from tui.parser import OutputParser, StepMetrics, FinalMetrics
from tui.resilience import Heartbeat
from tui.results import (
    ExperimentResult,
    append_result,
    format_history_for_prompt,
    get_best_result,
    init_results_tsv,
    load_results,
    next_experiment_number,
)


# Marker comments that delimit the hyperparameter block in train_mlx.py
HP_BLOCK_START = "# ---------------------------------------------------------------------------\n# Hyperparameters"
HP_BLOCK_END = "# ---------------------------------------------------------------------------\n# Setup"


@dataclass
class OrchestratorCallbacks:
    """Callbacks for communicating with the TUI.

    All callbacks are called from the orchestrator's background thread.
    The TUI app must wrap them with call_from_thread() for thread safety.
    """
    on_status_change: Callable[[str, str], None]       # (status, message)
    on_experiment_start: Callable[[int, str, str], None]  # (exp_num, desc, reasoning)
    on_training_output: Callable[[str], None]           # raw line from training stdout
    on_experiment_complete: Callable[[ExperimentResult], None]
    on_stats_update: Callable[[int, int, int, float], None]  # (total, kept, discarded, best_bpb)
    on_error: Callable[[str], None]


class ExperimentOrchestrator:
    """Manages the autonomous experiment loop.

    Runs in a dedicated background thread. Uses the LLM to generate
    code modifications, trains, evaluates, and decides keep/discard.

    Resilience: if the process dies and restarts, it reads results.tsv
    to determine the last completed experiment and resumes from there.
    """

    def __init__(
        self,
        training_script: str = "train_mlx.py",
        results_path: str = "results.tsv",
        max_experiments: int = 100,
        run_tag: str | None = None,
        callbacks: OrchestratorCallbacks | None = None,
        model: str | None = None,
        dataset_name: str = "",
    ):
        self._training_script = training_script
        self._results_path = results_path
        self._max_experiments = max_experiments
        self._run_tag = run_tag or time.strftime("%b%d").lower()
        self._callbacks = callbacks
        self._model_override = model  # e.g. "claude-sonnet-4-6"
        self._dataset_name = dataset_name

        self._git = GitManager()
        self._hw_info = get_hardware_summary()
        self._llm: LLMBackend | None = None  # lazy init

        # Heartbeat for external monitors
        heartbeat_dir = os.path.dirname(os.path.abspath(results_path)) if results_path else "."
        self._heartbeat = Heartbeat(os.path.join(heartbeat_dir, ".runner_status.json"))

        # State
        self._running = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

        # Counters
        self.total_runs = 0
        self.kept_count = 0
        self.discarded_count = 0
        self.crash_count = 0
        self.best_val_bpb = float("inf")
        self.best_experiment = "none"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the experiment loop in a background thread."""
        if self._running:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="orchestrator",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the orchestrator to stop after the current experiment."""
        self._stop_event.set()
        # Kill any running training subprocess
        with self._lock:
            if self._proc and self._proc.returncode is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()

    def cleanup(self) -> None:
        """Final cleanup — close heartbeat, etc."""
        self._heartbeat.close()

    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Callbacks (thread-safe dispatchers)
    # ------------------------------------------------------------------

    def _cb_status(self, status: str, message: str) -> None:
        if self._callbacks:
            self._callbacks.on_status_change(status, message)

    def _cb_experiment_start(self, exp_num: int, desc: str, reasoning: str) -> None:
        if self._callbacks:
            self._callbacks.on_experiment_start(exp_num, desc, reasoning)

    def _cb_training_output(self, line: str) -> None:
        if self._callbacks:
            self._callbacks.on_training_output(line)

    def _cb_experiment_complete(self, result: ExperimentResult) -> None:
        if self._callbacks:
            self._callbacks.on_experiment_complete(result)

    def _cb_stats_update(self) -> None:
        if self._callbacks:
            self._callbacks.on_stats_update(
                self.total_runs, self.kept_count,
                self.discarded_count, self.best_val_bpb,
            )

    def _cb_error(self, message: str) -> None:
        if self._callbacks:
            self._callbacks.on_error(message)

    def _update_heartbeat(self, experiment: int = 0, status: str = "running") -> None:
        """Update heartbeat with current state."""
        model_name = self._model_override or "default"
        self._heartbeat.update(
            experiment=experiment,
            status=status,
            dataset=self._dataset_name,
            best_bpb=self.best_val_bpb,
            model=model_name,
            total=self.total_runs,
            kept=self.kept_count,
            discarded=self.discarded_count,
            crashes=self.crash_count,
        )

    # ------------------------------------------------------------------
    # Resume and cleanup
    # ------------------------------------------------------------------

    def _restore_counters(self) -> None:
        """Restore kept/discarded/crash counters from existing results."""
        results = load_results(self._results_path)
        self.total_runs = len(results)
        self.kept_count = sum(1 for r in results if r.status in ("keep", "baseline"))
        self.discarded_count = sum(1 for r in results if r.status == "discard")
        self.crash_count = sum(1 for r in results if r.status == "crash")

    def _cleanup_interrupted_experiment(self) -> None:
        """Clean up state from a previously interrupted experiment.

        Handles two cases:
        1. Uncommitted changes (code modified, never committed) → restore file
        2. Orphaned commit (committed but training never finished) → revert
        """
        # Case 1: Dirty working tree
        if self._git.has_uncommitted_changes():
            self._cb_status("initializing", "Cleaning up uncommitted changes from interrupted run")
            try:
                self._git.reset_working_tree(self._training_script)
            except Exception as e:
                self._cb_error(f"Failed to clean working tree: {e}")

        # Case 2: Orphaned commit (committed but no results recorded)
        try:
            head_msg = self._git.head_commit_message()
            match = re.match(r"exp(\d+):", head_msg)
            if match:
                exp_num = int(match.group(1))
                results = load_results(self._results_path)
                recorded = {int(r.exp.replace("exp", ""))
                           for r in results if r.exp.startswith("exp")}
                if exp_num not in recorded:
                    self._cb_status("initializing",
                        f"Reverting orphaned commit: {head_msg[:60]}")
                    self._git.revert_last_commit()
        except Exception as e:
            self._cb_error(f"Orphan commit check failed: {e}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Main experiment loop (runs in background thread)."""
        self._running = True

        try:
            # Initialize LLM backend
            self._cb_status("initializing", "Connecting to LLM backend...")
            try:
                self._llm = get_llm_backend(model=self._model_override)
                self._cb_status("initializing", f"LLM: {self._llm.name()}")
            except Exception as e:
                self._cb_error(f"LLM backend error: {e}")
                return

            # Validate credentials with a preflight API call
            self._cb_status("initializing", "Validating API credentials...")
            try:
                if not self._llm.validate():
                    source = getattr(self._llm, '_cred_source', 'unknown')
                    self._cb_error(
                        f"API credential validation failed (source: {source}). "
                        f"Run: uv run dashboard.py --setup-key"
                    )
                    return
            except Exception as e:
                self._cb_error(f"API validation error: {e}")
                return
            self._cb_status("initializing", "API credentials validated ✓")

            # Set up branch
            branch_name = f"autoresearch/{self._run_tag}"
            current = self._git.current_branch()
            if current != branch_name:
                if self._git.branch_exists(branch_name):
                    self._cb_status("initializing", f"Checking out existing branch {branch_name}")
                    self._git.checkout(branch_name)
                else:
                    self._cb_status("initializing", f"Creating branch {branch_name}")
                    self._git.create_branch(branch_name)

            # Clean up any interrupted experiment from a previous run
            self._cleanup_interrupted_experiment()

            # Initialize results (validates existing file)
            init_results_tsv(self._results_path)

            # Load existing state (for resuming)
            existing_best, existing_exp = get_best_result(self._results_path)
            if existing_best < float("inf"):
                self.best_val_bpb = existing_best
                self.best_experiment = existing_exp

            # Restore counters from existing results
            self._restore_counters()

            start_exp = next_experiment_number(self._results_path)

            if start_exp > 0:
                self._cb_status("initializing",
                    f"Resuming from exp{start_exp} — "
                    f"{self.total_runs} completed, best: {self.best_val_bpb:.4f}")

            self._update_heartbeat(start_exp, "initializing")

            # Run baseline if this is a fresh start
            if start_exp == 0:
                self._cb_status("baseline", "Running baseline (no modifications)...")
                self._run_baseline()
                start_exp = 1

                if self._stop_event.is_set():
                    return

            # Main experiment loop
            # max_experiments is the TOTAL target, not "more from here"
            for exp_num in range(start_exp, self._max_experiments):
                if self._stop_event.is_set():
                    break

                self._update_heartbeat(exp_num, "running")
                self._run_experiment(exp_num)

                if self._stop_event.is_set():
                    break

                # Brief pause between experiments
                time.sleep(2)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self._cb_error(f"Orchestrator error: {e}")
            self._cb_error(f"Traceback: {tb}")
        finally:
            self._running = False
            self._update_heartbeat(status="stopped")
            self._cb_status("stopped", "Experiment loop stopped")

    # ------------------------------------------------------------------
    # Baseline run
    # ------------------------------------------------------------------

    def _run_baseline(self) -> None:
        """Run the training script without modifications to establish baseline."""
        self._cb_experiment_start(0, "baseline (no modifications)", "Establishing baseline with current defaults")
        self._update_heartbeat(0, "baseline")

        final = self._run_training()

        if final:
            result = ExperimentResult(
                exp="exp0",
                description="baseline (no modifications)",
                val_bpb=final.val_bpb,
                peak_mem_gb=final.peak_vram_mb / 1024,
                tok_sec=int(final.total_tokens_M * 1e6 / final.training_seconds) if final.training_seconds > 0 else 0,
                mfu=final.mfu_percent,
                steps=final.num_steps,
                status="baseline",
                notes=f"depth={final.depth}, {final.chip}",
            )
            self.best_val_bpb = final.val_bpb
            self.best_experiment = "exp0 (baseline)"
        else:
            result = ExperimentResult(
                exp="exp0",
                description="baseline (no modifications)",
                val_bpb=0.0, peak_mem_gb=0.0, tok_sec=0,
                mfu=0.0, steps=0, status="crash",
                notes="baseline training failed",
            )

        append_result(self._results_path, result)
        self.total_runs += 1
        if result.status == "baseline":
            self.kept_count += 1
        self._cb_experiment_complete(result)
        self._cb_stats_update()

    # ------------------------------------------------------------------
    # Single experiment
    # ------------------------------------------------------------------

    def _run_experiment(self, exp_num: int, _pause_depth: int = 0) -> None:
        """Run a single experiment: LLM → modify → commit → train → evaluate.

        If the API is completely unavailable after retries, pauses for 10
        minutes and retries (up to 3 pause cycles to avoid infinite loops).
        """

        # 1. Ask the LLM for a modification
        self._cb_status("thinking", f"Claude is designing experiment {exp_num}...")
        self._update_heartbeat(exp_num, "thinking")

        current_code = self._extract_hp_block()
        results_history = format_history_for_prompt(self._results_path)

        proposal = self._call_llm_with_backoff(
            current_code, results_history, exp_num
        )

        if proposal is None:
            # All retries exhausted — pause and retry if we haven't exceeded depth
            if _pause_depth < 3:
                self._cb_status("thinking",
                    f"API unavailable — pausing 10 min (attempt {_pause_depth + 1}/3)")
                self._update_heartbeat(exp_num, "paused_api_outage")
                for _ in range(120):  # 10 min in 5s chunks
                    if self._stop_event.is_set():
                        return
                    time.sleep(5)
                return self._run_experiment(exp_num, _pause_depth + 1)
            else:
                self._cb_error(f"Skipping exp{exp_num} — API unavailable after 30 min of retries")
                return

        self._cb_experiment_start(exp_num, proposal.description, proposal.reasoning)
        self._update_heartbeat(exp_num, "committing")

        # 2. Apply code changes
        self._cb_status("committing", f"Applying: {proposal.description}")
        try:
            self._apply_hp_block(proposal.code)
        except Exception as e:
            self._cb_error(f"Failed to apply code: {e}")
            return

        # Validate the modified code parses
        try:
            with open(self._training_script) as f:
                compile(f.read(), self._training_script, "exec")
        except SyntaxError as e:
            self._cb_error(f"Syntax error in modified code: {e}")
            # Restore original
            self._apply_hp_block(current_code)
            return

        # 3. Commit
        try:
            commit_hash = self._git.commit_changes(
                f"exp{exp_num}: {proposal.description}",
                [self._training_script],
            )
        except Exception as e:
            self._cb_error(f"Git commit failed: {e}")
            self._apply_hp_block(current_code)
            return

        # 4. Train
        self._cb_status("training", f"Training exp{exp_num}: {proposal.description}")
        self._update_heartbeat(exp_num, "training")
        final = self._run_training()

        # 5. Evaluate and decide
        self._cb_status("evaluating", "Comparing results...")
        self._update_heartbeat(exp_num, "evaluating")

        if final and final.val_bpb > 0:
            improved = final.val_bpb < self.best_val_bpb

            if improved:
                status = "keep"
                old_best = self.best_val_bpb
                self.best_val_bpb = final.val_bpb
                self.best_experiment = f"exp{exp_num} ({proposal.description})"
                self.kept_count += 1
                self._cb_status("evaluating",
                    f"KEEP — val_bpb improved: {final.val_bpb:.4f} (was {old_best:.4f})")
            else:
                status = "discard"
                self.discarded_count += 1
                self._cb_status("evaluating",
                    f"DISCARD — val_bpb {final.val_bpb:.4f} >= best {self.best_val_bpb:.4f}")
                self._git.revert_last_commit()

            result = ExperimentResult(
                exp=f"exp{exp_num}",
                description=proposal.description,
                val_bpb=final.val_bpb,
                peak_mem_gb=final.peak_vram_mb / 1024,
                tok_sec=int(final.total_tokens_M * 1e6 / final.training_seconds) if final.training_seconds > 0 else 0,
                mfu=final.mfu_percent,
                steps=final.num_steps,
                status=status,
                notes=proposal.reasoning[:80],
            )
        else:
            # Crash
            self.crash_count += 1
            self._cb_status("evaluating", "CRASH — training failed")
            self._git.revert_last_commit()

            result = ExperimentResult(
                exp=f"exp{exp_num}",
                description=proposal.description,
                val_bpb=0.0, peak_mem_gb=0.0, tok_sec=0,
                mfu=0.0, steps=0, status="crash",
                notes="training crashed or timed out",
            )

        append_result(self._results_path, result)
        self.total_runs += 1
        self._cb_experiment_complete(result)
        self._cb_stats_update()
        self._update_heartbeat(exp_num, "completed")

    # ------------------------------------------------------------------
    # LLM call with exponential backoff
    # ------------------------------------------------------------------

    def _call_llm_with_backoff(self, current_code, results_history, exp_num):
        """Call the LLM with exponential backoff for transient errors.

        Handles:
        - Rate limits (429): backoff 60s, 120s, 240s, 480s, 600s
        - Server errors (5xx): backoff 30s, 60s, 120s, 240s, 300s
        - Connection errors: backoff 30s, 60s, 120s, 240s, 300s
        - Other errors: backoff 10s, 20s, 30s, 40s, 50s

        Returns the proposal or None if all retries exhausted.
        """
        max_retries = 5

        for attempt in range(max_retries):
            try:
                proposal = self._llm.generate_experiment(
                    current_code=current_code,
                    results_history=results_history,
                    best_val_bpb=self.best_val_bpb,
                    best_experiment=self.best_experiment,
                    hw_info=self._hw_info,
                )
                return proposal

            except Exception as e:
                err_str = str(e).lower()
                err_type = type(e).__name__

                # Classify the error and determine backoff
                if "rate" in err_str or "429" in err_str:
                    wait = min(60 * (2 ** attempt), 600)
                    self._cb_status("thinking",
                        f"Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                elif "500" in err_str or "502" in err_str or "503" in err_str or "server" in err_str:
                    wait = min(30 * (2 ** attempt), 300)
                    self._cb_status("thinking",
                        f"API server error, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                elif "connection" in err_str or "timeout" in err_str:
                    wait = min(30 * (2 ** attempt), 300)
                    self._cb_status("thinking",
                        f"Connection error, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                else:
                    wait = 10 * (attempt + 1)
                    self._cb_error(
                        f"LLM error ({err_type}, attempt {attempt+1}/{max_retries}): {e}")

                self._update_heartbeat(exp_num, f"retrying_api_{attempt+1}")

                # Sleep in 5s chunks so stop_event can interrupt
                for _ in range(max(1, wait // 5)):
                    if self._stop_event.is_set():
                        return None
                    time.sleep(5)

        return None

    # ------------------------------------------------------------------
    # Training subprocess
    # ------------------------------------------------------------------

    def _run_training(self) -> Optional[FinalMetrics]:
        """Run the training script and return parsed final metrics.

        Returns None if the training crashed or timed out.
        """
        parser = OutputParser()

        python = sys.executable
        cmd = [python, "-u", self._training_script]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            with self._lock:
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env=env,
                    bufsize=0,
                )
        except Exception as e:
            self._cb_error(f"Failed to start training: {e}")
            return None

        # Read output byte-by-byte (same pattern as app.py)
        buffer = ""
        deadline = time.time() + 600  # 10 minute timeout

        try:
            while True:
                if self._stop_event.is_set():
                    self._proc.terminate()
                    self._proc.wait(timeout=5)
                    return None

                if time.time() > deadline:
                    self._cb_error("Training timed out (>10 min) — killing")
                    self._proc.kill()
                    self._proc.wait()
                    return None

                byte = self._proc.stdout.read(1)
                if not byte:
                    break

                char = byte.decode("utf-8", errors="replace")

                if char in ("\n", "\r"):
                    if buffer.strip():
                        results = parser.parse_line(buffer)
                        for item in results:
                            if isinstance(item, StepMetrics):
                                self._cb_training_output(buffer)
                            elif isinstance(item, str):
                                self._cb_training_output(item)
                    buffer = ""
                else:
                    buffer += char

        except Exception as e:
            self._cb_error(f"Error reading training output: {e}")
        finally:
            if buffer.strip():
                parser.parse_line(buffer)

            with self._lock:
                if self._proc:
                    self._proc.wait()
                    self._proc = None

        return parser.final

    # ------------------------------------------------------------------
    # Code manipulation
    # ------------------------------------------------------------------

    def _extract_hp_block(self) -> str:
        """Extract the hyperparameter block from train_mlx.py."""
        with open(self._training_script) as f:
            content = f.read()

        start_idx = content.find(HP_BLOCK_START)
        end_idx = content.find(HP_BLOCK_END)

        if start_idx == -1 or end_idx == -1:
            raise RuntimeError("Could not find hyperparameter block markers in training script")

        return content[start_idx:end_idx]

    def _apply_hp_block(self, new_block: str) -> None:
        """Replace the hyperparameter block in train_mlx.py."""
        with open(self._training_script) as f:
            content = f.read()

        start_idx = content.find(HP_BLOCK_START)
        end_idx = content.find(HP_BLOCK_END)

        if start_idx == -1 or end_idx == -1:
            raise RuntimeError("Could not find hyperparameter block markers in training script")

        # Ensure the new block ends with a newline
        if not new_block.endswith("\n"):
            new_block += "\n"

        # Ensure there's a blank line before the Setup marker
        if not new_block.endswith("\n\n"):
            new_block += "\n"

        new_content = content[:start_idx] + new_block + content[end_idx:]

        with open(self._training_script, "w") as f:
            f.write(new_content)
