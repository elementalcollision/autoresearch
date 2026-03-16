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
        self._vram_used_mb: float = 0

    def on_mount(self) -> None:
        self._refresh_content()

    def update_vram(self, vram_mb: float) -> None:
        self._vram_used_mb = vram_mb
        self._refresh_content()

    def _refresh_content(self) -> None:
        hw = self._hw
        total_gb = hw.get('total_memory_gb', 0)
        used_gb = self._vram_used_mb / 1024
        pct = (used_gb / total_gb * 100) if total_gb > 0 else 0

        bar_width = 20
        filled = int(bar_width * pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        lines = []
        lines.append(Text(hw.get('chip_name', 'Unknown'), style="bold cyan"))
        lines.append(Text(f"  Memory: {total_gb:.0f} GB unified", style="white"))

        if self._vram_used_mb > 0:
            lines.append(Text(
                f"  Used: {used_gb:.1f} / {total_gb:.0f} GB ({pct:.1f}%)",
                style="yellow" if pct > 75 else "green",
            ))
            lines.append(Text(f"  {bar}", style="yellow" if pct > 75 else "green"))
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
