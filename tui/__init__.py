"""Autoresearch TUI dashboard for Apple Silicon training monitoring."""


def __getattr__(name):
    """Lazy import DashboardApp to avoid requiring textual for headless use."""
    if name == "DashboardApp":
        from tui.app import DashboardApp
        return DashboardApp
    raise AttributeError(f"module 'tui' has no attribute {name!r}")


__all__ = ["DashboardApp"]
