#!/usr/bin/env python3
"""Launch the autoresearch TUI dashboard.

Usage:
    uv run dashboard.py                          # Single training run (default)
    uv run dashboard.py --agent                  # Autonomous experiment loop (requires ANTHROPIC_API_KEY)
    uv run dashboard.py --agent --tag mar16      # Custom run tag (default: today's date)
    uv run dashboard.py --agent --max 50         # Limit to 50 experiments (default: 100)
    uv run dashboard.py --watch                  # Watch mode (no training, monitor results.tsv)
    uv run dashboard.py train.py                 # Single run with MPS backend
"""

import sys


def main():
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print(__doc__)
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

    if "--tag" in args:
        idx = args.index("--tag")
        if idx + 1 < len(args):
            run_tag = args[idx + 1]
            args.pop(idx)  # remove --tag
            args.pop(idx)  # remove value
        else:
            print("Error: --tag requires a value")
            sys.exit(1)

    if "--max" in args:
        idx = args.index("--max")
        if idx + 1 < len(args):
            try:
                max_experiments = int(args[idx + 1])
            except ValueError:
                print("Error: --max requires an integer")
                sys.exit(1)
            args.pop(idx)
            args.pop(idx)
        else:
            print("Error: --max requires a value")
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
