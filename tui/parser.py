"""Parse training script stdout into structured metrics."""

import re
from dataclasses import dataclass, field


@dataclass
class StepMetrics:
    step: int = 0
    pct_done: float = 0.0
    loss: float = 0.0
    lrm: float = 0.0
    dt_ms: int = 0
    tok_per_sec: int = 0
    mfu: float = 0.0
    epoch: int = 0
    remaining: int = 0


@dataclass
class FinalMetrics:
    val_bpb: float = 0.0
    training_seconds: float = 0.0
    total_seconds: float = 0.0
    peak_vram_mb: float = 0.0
    mfu_percent: float = 0.0
    total_tokens_M: float = 0.0
    num_steps: int = 0
    num_params_M: float = 0.0
    depth: int = 0
    backend: str = ""
    chip: str = ""


# Matches: step 00192 (62.3%) | loss: 4.168331 | lrm: 0.66 | dt: 1001ms | tok/sec: 32,737 | mfu: 23.0% | epoch: 1 | remaining: 118s
STEP_RE = re.compile(
    r'step\s+(\d+)\s+\(([\d.]+)%\)\s+\|\s+'
    r'loss:\s+([\d.]+)\s+\|\s+'
    r'lrm:\s+([\d.]+)\s+\|\s+'
    r'dt:\s+(\d+)ms\s+\|\s+'
    r'tok/sec:\s+([\d,]+)\s+\|\s+'
    r'mfu:\s+([\d.]+)%\s+\|\s+'
    r'epoch:\s+(\d+)\s+\|\s+'
    r'remaining:\s+(\d+)s'
)

# Matches final summary lines like: val_bpb:          1.329263
FINAL_RE = re.compile(r'^(\w+):\s+(.+?)\s*$')


class OutputParser:
    """Parses training script stdout into structured data."""

    def __init__(self):
        self.startup_lines: list[str] = []
        self.in_final_block = False
        self._final_data: dict[str, str] = {}
        self.final: FinalMetrics | None = None

    def parse_line(self, raw_line: str) -> list[StepMetrics | str]:
        """Parse a line of output. Returns list of StepMetrics and/or log strings.

        Training output uses \\r to overwrite lines, so a single read may
        contain multiple step updates concatenated. We split on \\r to
        extract each one.
        """
        results = []

        # Split on \r to handle carriage-return-delimited step updates
        segments = raw_line.split('\r')

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            # Check for final summary separator
            if segment == '---':
                self.in_final_block = True
                results.append("--- Training complete. Running evaluation... ---")
                continue

            # Parse final summary block
            if self.in_final_block:
                m = FINAL_RE.match(segment)
                if m:
                    self._final_data[m.group(1)] = m.group(2)
                    # Check if we have all fields
                    if 'chip' in self._final_data:
                        self.final = self._build_final()
                        results.append(self._format_final())
                continue

            # Try to parse as step metrics
            m = STEP_RE.search(segment)
            if m:
                metrics = StepMetrics(
                    step=int(m.group(1)),
                    pct_done=float(m.group(2)),
                    loss=float(m.group(3)),
                    lrm=float(m.group(4)),
                    dt_ms=int(m.group(5)),
                    tok_per_sec=int(m.group(6).replace(',', '')),
                    mfu=float(m.group(7)),
                    epoch=int(m.group(8)),
                    remaining=int(m.group(9)),
                )
                results.append(metrics)
                continue

            # Startup/info line — log it
            if segment:
                self.startup_lines.append(segment)
                results.append(segment)

        return results

    def _build_final(self) -> FinalMetrics:
        d = self._final_data
        return FinalMetrics(
            val_bpb=float(d.get('val_bpb', 0)),
            training_seconds=float(d.get('training_seconds', 0)),
            total_seconds=float(d.get('total_seconds', 0)),
            peak_vram_mb=float(d.get('peak_vram_mb', 0)),
            mfu_percent=float(d.get('mfu_percent', 0)),
            total_tokens_M=float(d.get('total_tokens_M', 0)),
            num_steps=int(d.get('num_steps', 0)),
            num_params_M=float(d.get('num_params_M', 0)),
            depth=int(d.get('depth', 0)),
            backend=d.get('backend', ''),
            chip=d.get('chip', ''),
        )

    def _format_final(self) -> str:
        f = self.final
        return (
            f"Evaluation complete: val_bpb={f.val_bpb:.4f} | "
            f"peak_vram={f.peak_vram_mb:.0f}MB | "
            f"MFU={f.mfu_percent:.1f}% | "
            f"steps={f.num_steps} | "
            f"tokens={f.total_tokens_M:.1f}M"
        )
