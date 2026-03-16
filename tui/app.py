"""Main Textual Application for the autoresearch dashboard."""

import os
import subprocess
import sys
import threading
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer

from tui.hardware import get_hardware_summary
from tui.parser import OutputParser, StepMetrics
from tui.widgets import TrainingPanel, HardwarePanel, ExperimentsTable, ActivityLog


class DashboardApp(App):
    """Autoresearch training dashboard for Apple Silicon."""

    TITLE = "autoresearch"
    SUB_TITLE = "Apple Silicon Training Dashboard"
    CSS_PATH = "styles.tcss"

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Dark/Light"),
        ("r", "reload_experiments", "Reload"),
    ]

    def __init__(self, training_script: str = "train_mlx.py", **kwargs):
        super().__init__(**kwargs)
        self._training_script = training_script
        self._hw_info = get_hardware_summary()
        self._proc: subprocess.Popen | None = None
        self._parser = OutputParser()
        self._reader_thread: threading.Thread | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="top-row"):
            with Vertical(id="training-panel") as v:
                v.border_title = "Training"
                yield TrainingPanel(id="training")
            with Vertical(id="hardware-panel") as v:
                v.border_title = "Hardware"
                yield HardwarePanel(self._hw_info, id="hardware")
        with Vertical(id="experiments-panel") as v:
            v.border_title = "Experiments"
            yield ExperimentsTable(id="experiments")
        with Vertical(id="activity-panel") as v:
            v.border_title = "Activity Log"
            yield ActivityLog(id="activity")
        yield Footer()

    async def on_mount(self) -> None:
        log = self.query_one("#activity", ActivityLog)
        training = self.query_one("#training", TrainingPanel)

        log.log_message(f"Dashboard started — {self._hw_info.get('chip_name', 'Unknown')}")
        log.log_message(f"Training script: {self._training_script}")

        if self._training_script == "__watch__":
            training.set_description("Watch mode — no training")
            return

        if not os.path.exists(self._training_script):
            log.log_message(f"Script not found: {self._training_script}", style="bold red")
            training.set_description(f"Error: {self._training_script} not found")
            return

        self._start_training()

    def _start_training(self) -> None:
        """Launch training subprocess and reader thread."""
        log = self.query_one("#activity", ActivityLog)
        training = self.query_one("#training", TrainingPanel)

        python = sys.executable
        cmd = [python, "-u", self._training_script]  # -u for unbuffered

        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'

        log.log_message(f"Launching: {' '.join(cmd)}")

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,  # Prevent stdin conflicts with Textual
                env=env,
                bufsize=0,  # unbuffered pipe
            )
        except Exception as e:
            log.log_message(f"Failed to start: {e}", style="bold red")
            training.set_description(f"Error: {e}")
            return

        log.log_message("Process started, reading output...")
        training.set_description("Compiling model (first step may take 30-60s)...")

        # Reader thread reads byte-by-byte and uses call_from_thread to
        # dispatch lines to the main Textual event loop. This avoids the
        # asyncio.Queue + call_soon_threadsafe pattern which can fail when
        # Textual manages its own event loop.
        self._reader_thread = threading.Thread(
            target=self._reader_worker,
            args=(self._proc,),
            daemon=True,
        )
        self._reader_thread.start()

    def _reader_worker(self, proc: subprocess.Popen) -> None:
        """Thread that reads subprocess stdout byte-by-byte to handle \\r updates.

        Training scripts use print(..., end="", flush=True) with \\r for in-place
        line updates. We read byte-by-byte to capture each flush immediately,
        splitting on \\r and \\n to extract individual step updates.

        Each complete line is dispatched to the main Textual thread via
        call_from_thread, which is Textual's thread-safe callback mechanism.
        """
        buffer = ""
        try:
            while True:
                byte = proc.stdout.read(1)
                if not byte:
                    break
                char = byte.decode('utf-8', errors='replace')

                if char == '\n':
                    if buffer.strip():
                        line = buffer
                        self.call_from_thread(self._on_training_output, line)
                    buffer = ""
                elif char == '\r':
                    if buffer.strip():
                        line = buffer
                        self.call_from_thread(self._on_training_output, line)
                    buffer = ""
                else:
                    buffer += char
        except Exception:
            pass
        finally:
            if buffer.strip():
                line = buffer
                self.call_from_thread(self._on_training_output, line)
            # Signal completion
            self.call_from_thread(self._on_training_done, proc.wait())

    def _on_training_output(self, line: str) -> None:
        """Process a line of training output on the main thread."""
        training = self.query_one("#training", TrainingPanel)
        hardware = self.query_one("#hardware", HardwarePanel)
        log = self.query_one("#activity", ActivityLog)

        results = self._parser.parse_line(line)

        for item in results:
            if isinstance(item, StepMetrics):
                training.update_metrics(item)
            elif isinstance(item, str):
                log.log_message(item)

                # Detect backend from startup line
                if item.startswith("Backend:"):
                    backend = item.split("(")[0].replace("Backend:", "").strip()
                    training.set_backend(backend)

                # Update VRAM from final output
                if "peak_vram" in item and self._parser.final:
                    hardware.update_vram(self._parser.final.peak_vram_mb)
                    training.update_final(self._parser.final)

    def _on_training_done(self, returncode: int) -> None:
        """Handle training subprocess completion on the main thread."""
        log = self.query_one("#activity", ActivityLog)
        self._proc = None

        if returncode == 0:
            log.log_message("Training process exited successfully.", style="bold green")
        else:
            log.log_message(f"Training process exited with code {returncode}.", style="bold red")

        self.action_reload_experiments()

    def action_reload_experiments(self) -> None:
        """Reload the experiments table from results.tsv."""
        table = self.query_one("#experiments", ExperimentsTable)
        table.load_data()
        log = self.query_one("#activity", ActivityLog)
        log.log_message("Experiments table reloaded.")

    async def _on_exit(self) -> None:
        """Kill training subprocess on exit."""
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
