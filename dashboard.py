#!/usr/bin/env python3
"""Launch the autoresearch TUI dashboard.

Usage:
    uv run dashboard.py                          # Single training run (default)
    uv run dashboard.py --agent                  # Autonomous experiment loop
    uv run dashboard.py --agent --tag mar16      # Custom run tag (default: today's date)
    uv run dashboard.py --agent --max 50         # Limit to 50 experiments (default: 100)
    uv run dashboard.py --watch                  # Watch mode (no training, monitor results.tsv)
    uv run dashboard.py train.py                 # Single run with MPS backend

Credential management:
    uv run dashboard.py --setup-key              # Store API key in macOS Keychain (one-time)
    uv run dashboard.py --clear-key              # Remove stored API key from Keychain
    uv run dashboard.py --check-key              # Check which credential source is active
"""

import sys


def _pop_arg(args, flag):
    """Remove a flag from args and return its value. Handles --flag=value and --flag value."""
    # Check --flag=value format first
    for i, a in enumerate(args):
        if a.startswith(f"{flag}="):
            args.pop(i)
            return a.split("=", 1)[1]
    # Check --flag value format
    if flag in args:
        idx = args.index(flag)
        if idx + 1 < len(args):
            val = args[idx + 1]
            args.pop(idx)  # remove flag
            args.pop(idx)  # remove value
            return val
    return None


def main():
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print(__doc__)
        sys.exit(0)

    # Credential management commands (non-TUI, exit after)
    if "--setup-key" in args:
        from tui.credentials import setup_api_key
        setup_api_key()
        sys.exit(0)

    if "--clear-key" in args:
        from tui.credentials import clear_api_key
        clear_api_key()
        sys.exit(0)

    if "--check-key" in args:
        from tui.credentials import resolve_api_key
        try:
            cred = resolve_api_key()
            masked = cred.api_key[:12] + "..." + cred.api_key[-4:]
            source_desc = {
                "env": "ANTHROPIC_API_KEY environment variable",
                "keychain": "macOS Keychain (autoresearch-agent)",
                "claude-code": "Claude Code credentials (OAuth token — may not work with API)",
            }
            print(f"Active credential: {masked}")
            print(f"Source: {source_desc.get(cred.source, cred.source)}")
            if cred.source == "claude-code":
                print()
                print("Note: Claude Code OAuth tokens may not work with the Anthropic API.")
                print("For reliable agent mode, store a proper API key:")
                print("  uv run dashboard.py --setup-key")
        except RuntimeError as e:
            print(str(e))
            sys.exit(1)
        sys.exit(0)

    # Parse flags
    mode = "single"
    max_experiments = 100
    run_tag = None
    training_script = "train_mlx.py"

    if "--watch" in args:
        mode = "watch"
        args.remove("--watch")

    if "--agent" in args:
        mode = "agent"
        args.remove("--agent")

    tag_val = _pop_arg(args, "--tag")
    if tag_val is not None:
        run_tag = tag_val

    max_val = _pop_arg(args, "--max")
    if max_val is not None:
        try:
            max_experiments = int(max_val)
        except ValueError:
            print("Error: --max requires an integer")
            sys.exit(1)

    # Remaining positional arg is the training script
    if args:
        training_script = args[0]

    from tui.app import DashboardApp

    if mode == "watch":
        app = DashboardApp(training_script="__watch__", mode="watch")
    elif mode == "agent":
        app = DashboardApp(
            training_script=training_script,
            mode="agent",
            max_experiments=max_experiments,
            run_tag=run_tag,
        )
    else:
        app = DashboardApp(training_script=training_script, mode="single")

    app.run()


if __name__ == "__main__":
    main()
