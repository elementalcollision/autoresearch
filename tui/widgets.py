"""Custom Textual widgets for the autoresearch dashboard."""

from textual.widgets import Static, DataTable, RichLog
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.progress_bar import ProgressBar

from tui.parser import StepMetrics, FinalMetrics
from tui.experiments import load_experiments


class TrainingPanel(Static):
    """Displays real-time training progress."""

    DEFAULT_CSS = """
    TrainingPanel {
        height: 100%;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._metrics: StepMetrics | None = None
        self._final: FinalMetrics | None = None
        self._description: str = "Waiting for training..."
        self._backend: str = ""

    def on_mount(self) -> None:
        self._refresh_content()

    def set_description(self, desc: str) -> None:
        self._description = desc
        self._refresh_content()

    def set_backend(self, backend: str) -> None:
        self._backend = backend
        self._refresh_content()

    def update_metrics(self, metrics: StepMetrics) -> None:
        self._metrics = metrics
        self._refresh_content()

    def update_final(self, final: FinalMetrics) -> None:
        self._final = final
        self._refresh_content()

    def _refresh_content(self) -> None:
        if self._final:
            self._show_final()
        elif self._metrics:
            self._show_training()
        else:
            self.update(Text(self._description, style="dim"))

    def _show_training(self) -> None:
        m = self._metrics
        pct = m.pct_done / 100.0

        # Build progress bar
        bar_width = 30
        filled = int(bar_width * pct)
        bar = "█" * filled + "░" * (bar_width - filled)

        lines = []
        lines.append(Text(self._description, style="bold cyan"))
        lines.append(Text(f"  {bar} {m.pct_done:5.1f}%", style="green"))
        lines.append(Text(
            f"  Step: {m.step:05d}  │  ETA: {m.remaining}s",
            style="white",
        ))
        lines.append(Text(
            f"  Loss: {m.loss:.4f}  │  LR: {m.lrm:.2f}",
            style="white",
        ))
        lines.append(Text(
            f"  Tok/sec: {m.tok_per_sec:,}  │  MFU: {m.mfu:.1f}%",
            style="yellow",
        ))

        backend_str = self._backend or "detecting..."
        lines.append(Text(
            f"  Epoch: {m.epoch}  │  Backend: {backend_str}",
            style="dim",
        ))

        self.update(Text("\n").join(lines))

    def _show_final(self) -> None:
        f = self._final
        lines = []
        lines.append(Text("Training Complete", style="bold green"))
        lines.append(Text(f"  val_bpb: {f.val_bpb:.4f}", style="bold white"))
        lines.append(Text(
            f"  Peak VRAM: {f.peak_vram_mb:.0f} MB  │  MFU: {f.mfu_percent:.1f}%",
            style="white",
        ))
        lines.append(Text(
            f"  Steps: {f.num_steps}  │  Tokens: {f.total_tokens_M:.1f}M",
            style="white",
        ))
        lines.append(Text(
            f"  Time: {f.training_seconds:.0f}s train / {f.total_seconds:.0f}s total",
            style="dim",
        ))
        lines.append(Text(
            f"  Model: {f.num_params_M:.1f}M params, depth={f.depth}",
            style="dim",
        ))

        self.update(Text("\n").join(lines))


class HardwarePanel(Static):
    """Displays Apple Silicon hardware info and memory usage."""

    DEFAULT_CSS = """
    HardwarePanel {
        height: 100%;
        padding: 0 1;
    }
    """

    def __init__(self, hw_info: dict, **kwargs):
        super().__init__(**kwargs)
        self._hw = hw_info
        self._live_mb: float = 0       # current RSS from polling
        self._peak_mb: float = 0       # peak from final output or max(live)
        self._live_peak_mb: float = 0  # highest live reading this run

    def on_mount(self) -> None:
        self._refresh_content()

    def update_vram(self, vram_mb: float) -> None:
        """Update with final peak VRAM from training output."""
        self._peak_mb = vram_mb
        self._refresh_content()

    def update_live_memory(self, rss_mb: float) -> None:
        """Update with live RSS from subprocess polling."""
        self._live_mb = rss_mb
        if rss_mb > self._live_peak_mb:
            self._live_peak_mb = rss_mb
        self._refresh_content()

    def reset_memory(self) -> None:
        """Reset memory tracking for a new experiment run."""
        self._live_mb = 0
        self._live_peak_mb = 0
        self._peak_mb = 0
        self._refresh_content()

    def _refresh_content(self) -> None:
        hw = self._hw
        total_gb = hw.get('total_memory_gb', 0)

        # Use live reading if available, otherwise peak from final output
        display_mb = self._live_mb if self._live_mb > 0 else self._peak_mb
        display_gb = display_mb / 1024
        pct = (display_gb / total_gb * 100) if total_gb > 0 else 0

        bar_width = 20
        filled = int(bar_width * pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        lines = []
        lines.append(Text(hw.get('chip_name', 'Unknown'), style="bold cyan"))
        lines.append(Text(f"  Memory: {total_gb:.0f} GB unified", style="white"))

        if display_mb > 0:
            mem_style = "red" if pct > 90 else ("yellow" if pct > 75 else "green")
            lines.append(Text(
                f"  Used: {display_gb:.1f} / {total_gb:.0f} GB ({pct:.1f}%)",
                style=mem_style,
            ))
            lines.append(Text(f"  {bar}", style=mem_style))

            # Show peak if different from current
            peak_display = max(self._live_peak_mb, self._peak_mb)
            if peak_display > 0 and abs(peak_display - display_mb) > 100:
                lines.append(Text(
                    f"  Peak: {peak_display / 1024:.1f} GB",
                    style="dim",
                ))
        else:
            lines.append(Text("  Used: waiting...", style="dim"))
            lines.append(Text(f"  {'░' * bar_width}", style="dim"))

        cores = hw.get('gpu_cores', 0)
        tflops = hw.get('peak_tflops', 0)
        lines.append(Text(f"  GPU Cores: {cores}", style="white"))
        lines.append(Text(f"  Peak: {tflops:.1f} TFLOPS bf16", style="dim"))

        self.update(Text("\n").join(lines))


class ExperimentsTable(DataTable):
    """Displays experiment history from results.tsv."""

    DEFAULT_CSS = """
    ExperimentsTable {
        height: 100%;
    }
    """

    def __init__(self, tsv_path: str = "results.tsv", **kwargs):
        super().__init__(**kwargs)
        self._tsv_path = tsv_path

    def on_mount(self) -> None:
        self.add_columns("Exp", "Status", "val_bpb", "Mem(GB)", "Tok/s", "MFU", "Steps", "Description")
        self.load_data()

    def load_data(self) -> None:
        self.clear()
        experiments = load_experiments(self._tsv_path)

        for exp in experiments:
            status = exp.status.lower()
            if status == "keep":
                style = "green"
            elif status == "discard":
                style = "red"
            elif status == "baseline":
                style = "cyan"
            else:
                style = "white"

            self.add_row(
                Text(exp.exp, style=style),
                Text(exp.status, style=style),
                Text(exp.val_bpb, style="bold" if status == "keep" else ""),
                Text(exp.peak_mem_gb),
                Text(exp.tok_sec),
                Text(exp.mfu),
                Text(exp.steps),
                Text(exp.description[:40], style="dim"),
            )


class ExperimentStatusPanel(Static):
    """Displays autonomous experiment loop status."""

    DEFAULT_CSS = """
    ExperimentStatusPanel {
        height: 100%;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._status = "idle"
        self._message = "Waiting to start..."
        self._exp_num = 0
        self._total_runs = 0
        self._kept = 0
        self._discarded = 0
        self._best_bpb = float("inf")
        self._description = ""
        self._reasoning = ""

    def on_mount(self) -> None:
        self._refresh_content()

    def update_status(self, status: str, message: str) -> None:
        self._status = status
        self._message = message
        self._refresh_content()

    def update_stats(self, total: int, kept: int, discarded: int, best_bpb: float) -> None:
        self._total_runs = total
        self._kept = kept
        self._discarded = discarded
        self._best_bpb = best_bpb
        self._refresh_content()

    def set_experiment_info(self, exp_num: int, description: str, reasoning: str) -> None:
        self._exp_num = exp_num
        self._description = description
        self._reasoning = reasoning
        self._refresh_content()

    def _refresh_content(self) -> None:
        status_colors = {
            "idle": "dim",
            "initializing": "cyan",
            "baseline": "cyan",
            "thinking": "yellow",
            "committing": "white",
            "training": "green",
            "evaluating": "magenta",
            "stopped": "red",
        }
        status_style = status_colors.get(self._status, "white")

        lines = []

        # Status line
        status_icon = {
            "idle": "◯", "initializing": "⟳", "baseline": "▶",
            "thinking": "🧠", "committing": "📝", "training": "▶",
            "evaluating": "📊", "stopped": "■",
        }.get(self._status, "•")
        lines.append(Text(
            f"  {status_icon} {self._status.upper()}: {self._message}",
            style=f"bold {status_style}",
        ))

        # Stats line
        best_str = f"{self._best_bpb:.4f}" if self._best_bpb < float("inf") else "—"
        lines.append(Text(
            f"  Runs: {self._total_runs}  │  Kept: {self._kept}  │  "
            f"Discarded: {self._discarded}  │  Best: {best_str}",
            style="white",
        ))

        # Current experiment info
        if self._description:
            lines.append(Text(
                f"  Exp {self._exp_num}: {self._description[:60]}",
                style="cyan",
            ))

        self.update(Text("\n").join(lines))


class ActivityLog(RichLog):
    """Scrollable activity log."""

    DEFAULT_CSS = """
    ActivityLog {
        height: 100%;
        scrollbar-size: 1 1;
    }
    """

    def log_message(self, message: str, style: str = "") -> None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        if style:
            self.write(Text(f"[{timestamp}] {message}", style=style))
        else:
            self.write(f"[{timestamp}] {message}")
