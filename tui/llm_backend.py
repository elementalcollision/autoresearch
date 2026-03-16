"""LLM backend abstraction for generating experiment modifications.

Supports Claude API (Option A) with a placeholder for local LLMs via Ollama (Option B).
"""

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ExperimentProposal:
    """A proposed code modification from the LLM."""
    description: str  # one-line description for results.tsv
    reasoning: str    # 2-3 sentences explaining the hypothesis
    code: str         # replacement code for the hyperparameter block


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an autonomous AI researcher optimizing a small language model on Apple Silicon.

You modify the hyperparameter block of a training script to minimize val_bpb (validation bits per byte — lower is better). Each experiment runs for a fixed 5-minute time budget.

Rules:
- You may ONLY modify the hyperparameter block shown between the marker comments.
- You may change: batch sizes, learning rates, weight decay, warmup/warmdown ratios, model depth, aspect ratio, head dim, window pattern, MLP ratio, or any constant in that block.
- You may NOT add imports, modify the model class, optimizer, data loading, or evaluation.
- Make ONE change per experiment. This isolates the effect and makes results interpretable.
- Consider the full results history — don't repeat failed experiments.
- If many experiments have been discarded, try a different direction entirely.
- The key insight from prior characterization: maximizing gradient steps within the fixed time budget is the dominant factor. Smaller batches = more steps = usually better, up to a point.

Respond in EXACTLY this format (no markdown fences around the whole response):

DESCRIPTION: <one-line description, e.g. "Increase MATRIX_LR from 0.04 to 0.06">
REASONING: <2-3 sentences explaining why this change might improve val_bpb>
CODE:
<the complete replacement hyperparameter block, from the opening marker comment to the closing marker comment, inclusive>
"""

USER_PROMPT_TEMPLATE = """\
Here is the current hyperparameter block of train_mlx.py:

```python
{current_code}
```

Here is the experiment history so far:

{results_history}

Hardware: {chip_name}, {memory_gb:.0f} GB unified memory, {gpu_cores} GPU cores, ~{peak_tflops:.1f} TFLOPS bf16

Current best val_bpb: {best_val_bpb:.6f} (from {best_experiment})

Propose the next experiment. Remember: ONE change, and respond in the exact format specified.
"""


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_llm_response(response_text: str) -> ExperimentProposal:
    """Parse the LLM response into an ExperimentProposal.

    Expected format:
        DESCRIPTION: ...
        REASONING: ...
        CODE:
        <code block>
    """
    # Extract DESCRIPTION
    desc_match = re.search(r'^DESCRIPTION:\s*(.+?)$', response_text, re.MULTILINE)
    if not desc_match:
        raise ValueError("Response missing DESCRIPTION field")
    description = desc_match.group(1).strip()

    # Extract REASONING
    reason_match = re.search(r'^REASONING:\s*(.+?)(?=\nCODE:)', response_text, re.MULTILINE | re.DOTALL)
    if not reason_match:
        raise ValueError("Response missing REASONING field")
    reasoning = reason_match.group(1).strip()

    # Extract CODE — everything after "CODE:" line
    code_match = re.search(r'^CODE:\s*\n(.*)', response_text, re.MULTILINE | re.DOTALL)
    if not code_match:
        raise ValueError("Response missing CODE field")
    code = code_match.group(1).strip()

    # Strip markdown code fences if present
    code = re.sub(r'^```(?:python)?\s*\n', '', code)
    code = re.sub(r'\n```\s*$', '', code)

    return ExperimentProposal(
        description=description,
        reasoning=reasoning,
        code=code,
    )


# ---------------------------------------------------------------------------
# Backend ABC
# ---------------------------------------------------------------------------

class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate_experiment(
        self,
        current_code: str,
        results_history: str,
        best_val_bpb: float,
        best_experiment: str,
        hw_info: dict,
    ) -> ExperimentProposal:
        """Generate a proposed code modification."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
        ...


# ---------------------------------------------------------------------------
# Claude Backend (Option A)
# ---------------------------------------------------------------------------

class ClaudeBackend(LLMBackend):
    """Claude API backend using the Anthropic SDK."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: uv sync --extra agent"
            )

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def name(self) -> str:
        return f"Claude ({self._model})"

    def generate_experiment(
        self,
        current_code: str,
        results_history: str,
        best_val_bpb: float,
        best_experiment: str,
        hw_info: dict,
    ) -> ExperimentProposal:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            current_code=current_code,
            results_history=results_history if results_history else "No experiments yet — this will be the first modification after baseline.",
            chip_name=hw_info.get("chip_name", "Unknown"),
            memory_gb=hw_info.get("total_memory_gb", 0),
            gpu_cores=hw_info.get("gpu_cores", 0),
            peak_tflops=hw_info.get("peak_tflops", 0),
            best_val_bpb=best_val_bpb,
            best_experiment=best_experiment,
        )

        response = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        response_text = response.content[0].text
        return parse_llm_response(response_text)


# ---------------------------------------------------------------------------
# Ollama Backend (Option B — placeholder)
# ---------------------------------------------------------------------------

class OllamaBackend(LLMBackend):
    """Local LLM backend via Ollama (placeholder for future implementation).

    To use, set OLLAMA_MODEL environment variable (e.g., 'qwen2.5-coder:32b').
    Future implementation will use the Ollama HTTP API or litellm.
    """

    def __init__(self):
        self._model = os.environ.get("OLLAMA_MODEL", "")

    def name(self) -> str:
        return f"Ollama ({self._model})"

    def generate_experiment(
        self,
        current_code: str,
        results_history: str,
        best_val_bpb: float,
        best_experiment: str,
        hw_info: dict,
    ) -> ExperimentProposal:
        raise NotImplementedError(
            f"Ollama backend ({self._model}) is not yet implemented. "
            "Set ANTHROPIC_API_KEY to use Claude, or contribute an Ollama "
            "implementation to tui/llm_backend.py."
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_llm_backend() -> LLMBackend:
    """Create the appropriate LLM backend based on environment variables.

    Priority:
    1. ANTHROPIC_API_KEY → ClaudeBackend
    2. OLLAMA_MODEL → OllamaBackend (placeholder)
    3. Error
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ClaudeBackend()
    elif os.environ.get("OLLAMA_MODEL"):
        return OllamaBackend()
    else:
        raise RuntimeError(
            "No LLM backend configured. Set one of:\n"
            "  ANTHROPIC_API_KEY=sk-ant-...  (Claude API)\n"
            "  OLLAMA_MODEL=qwen2.5-coder:32b  (local, not yet implemented)"
        )
