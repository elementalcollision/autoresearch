"""Main Textual Application for the autoresearch dashboard."""

import asyncio
import os
import sys
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
        self._process: asyncio.subprocess.Process | None = None
        self._parser = OutputParser()

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

        # Check if script exists
        if not os.path.exists(self._training_script):
            log.log_message(f"Script not found: {self._training_script}", style="bold red")
            training.set_description(f"Error: {self._training_script} not found")
            return

        # Start training subprocess
        self._start_training()

    def _start_training(self) -> None:
        """Launch training as an async background task."""
        self.run_worker(self._run_training(), exclusive=True)

    async def _run_training(self) -> None:
        log = self.query_one("#activity", ActivityLog)
        training = self.query_one("#training", TrainingPanel)
        hardware = self.query_one("#hardware", HardwarePanel)

        training.set_description("Starting training...")

        # Find Python executable — prefer uv run if available
        python = sys.executable
        cmd = [python, self._training_script]

        # Set up environment to ensure unbuffered output
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'

        log.log_message(f"Launching: {' '.join(cmd)}")

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
        except Exception as e:
            log.log_message(f"Failed to start: {e}", style="bold red")
            training.set_description(f"Error: {e}")
            return

        log.log_message("Process started, waiting for output...")

        # Read stdout line by line
        assert self._process.stdout is not None
        buffer = ""

        while True:
            # Read chunks since training uses \r for in-place updates
            chunk = await self._process.stdout.read(4096)
            if not chunk:
                break

            text = chunk.decode('utf-8', errors='replace')
            buffer += text

            # Process complete segments (split on \n for line-based output)
            # But also handle \r-only updates within a line
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                self._process_line(line, training, hardware, log)

            # If buffer has \r segments but no \n, process the latest step update
            if '\r' in buffer:
                segments = buffer.split('\r')
                # Keep only the last segment as the active buffer
                for seg in segments[:-1]:
                    if seg.strip():
                        self._process_line(seg, training, hardware, log)
                buffer = segments[-1]

        # Process remaining buffer
        if buffer.strip():
            self._process_line(buffer, training, hardware, log)

        # Wait for process to finish
        returncode = await self._process.wait()
        self._process = None

        if returncode == 0:
            log.log_message("Training process exited successfully.", style="bold green")
        else:
            log.log_message(f"Training process exited with code {returncode}.", style="bold red")

        # Reload experiments table in case results.tsv was updated
        self.action_reload_experiments()

    def _process_line(
        self,
        line: str,
        training: TrainingPanel,
        hardware: HardwarePanel,
        log: ActivityLog,
    ) -> None:
        """Process a single line/chunk of training output."""
        results = self._parser.parse_line(line)

        for item in results:
            if isinstance(item, StepMetrics):
                training.update_metrics(item)
                # Update VRAM from peak_vram if we have final metrics
                # (live VRAM not available without importing ML frameworks)
            elif isinstance(item, str):
                log.log_message(item)

                # Detect backend from startup line
                if item.startswith("Backend:"):
                    backend = item.split("(")[0].replace("Backend:", "").strip()
                    training.set_backend(backend)

                # Update VRAM from peak memory line in final output
                if "peak_vram" in item and self._parser.final:
                    hardware.update_vram(self._parser.final.peak_vram_mb)
                    training.update_final(self._parser.final)

    def action_reload_experiments(self) -> None:
        """Reload the experiments table from results.tsv."""
        table = self.query_one("#experiments", ExperimentsTable)
        table.load_data()
        log = self.query_one("#activity", ActivityLog)
        log.log_message("Experiments table reloaded.")

    async def _on_exit(self) -> None:
        """Kill training subprocess on exit."""
        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
