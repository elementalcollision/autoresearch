"""Git operations for the experiment orchestrator."""

import subprocess
from pathlib import Path


class GitManager:
    """Manages git operations for the experiment loop.

    All operations run synchronously (designed to be called from the
    orchestrator's background thread, not the Textual event loop).
    """

    def __init__(self, repo_path: str = "."):
        self._repo = Path(repo_path).resolve()

    def _run(self, *args: str, check: bool = True) -> str:
        """Run a git command and return stdout."""
        result = subprocess.run(
            ["git"] + list(args),
            cwd=self._repo,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed (rc={result.returncode}): "
                f"{result.stderr.strip()}"
            )
        return result.stdout.strip()

    def current_branch(self) -> str:
        """Get the current branch name."""
        return self._run("rev-parse", "--abbrev-ref", "HEAD")

    def current_commit(self) -> str:
        """Get the current short commit hash."""
        return self._run("rev-parse", "--short=7", "HEAD")

    def branch_exists(self, branch: str) -> bool:
        """Check if a branch already exists (local or remote)."""
        result = self._run("branch", "--list", branch, check=False)
        if result:
            return True
        result = self._run("branch", "-r", "--list", f"origin/{branch}", check=False)
        return bool(result)

    def create_branch(self, branch_name: str) -> str:
        """Create and checkout a new branch. Returns branch name."""
        self._run("checkout", "-b", branch_name)
        return branch_name

    def checkout(self, branch: str) -> None:
        """Checkout an existing branch."""
        self._run("checkout", branch)

    def commit_changes(self, message: str, files: list[str]) -> str:
        """Stage files and commit. Returns short commit hash."""
        for f in files:
            self._run("add", f)
        self._run("commit", "-m", message)
        return self.current_commit()

    def revert_last_commit(self) -> None:
        """Hard-reset to the previous commit (discard last experiment)."""
        self._run("reset", "--hard", "HEAD~1")

    def read_file(self, path: str) -> str:
        """Read a file from the working tree."""
        full_path = self._repo / path
        return full_path.read_text()

    def write_file(self, path: str, content: str) -> None:
        """Write content to a file in the working tree."""
        full_path = self._repo / path
        full_path.write_text(content)

    def has_uncommitted_changes(self) -> bool:
        """Check if there are uncommitted changes."""
        status = self._run("status", "--porcelain", check=False)
        # Filter out untracked files like run.log
        for line in status.splitlines():
            if not line.startswith("??"):
                return True
        return False

    def reset_working_tree(self, file_path: str) -> None:
        """Restore a single file to its committed state."""
        self._run("checkout", "--", file_path)

    def head_commit_message(self) -> str:
        """Get the commit message of HEAD."""
        return self._run("log", "--format=%s", "-1")
