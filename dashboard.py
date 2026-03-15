#!/usr/bin/env python3
"""Launch the autoresearch TUI dashboard.

Usage:
    uv run dashboard.py                    # defaults to train_mlx.py
    uv run dashboard.py train.py           # use MPS backend
    uv run dashboard.py train_mlx.py       # explicit MLX backend
    uv run dashboard.py --watch            # watch mode (no training, monitor results.tsv)
"""

import sys


def main():
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print(__doc__)
        sys.exit(0)

    watch_mode = "--watch" in args
    if watch_mode:
        args.remove("--watch")

    training_script = args[0] if args else "train_mlx.py"

    from tui.app import DashboardApp

    if watch_mode:
        # Watch mode: show dashboard without starting training
        app = DashboardApp(training_script="__watch__")
    else:
        app = DashboardApp(training_script=training_script)

    app.run()


if __name__ == "__main__":
    main()
