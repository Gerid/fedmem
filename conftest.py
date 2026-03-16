from __future__ import annotations

import os
import pathlib

import matplotlib
import pytest

# ---------------------------------------------------------------------------
# 1. Force non-interactive matplotlib backend (no GUI windows during tests).
# ---------------------------------------------------------------------------
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# 2. Keep matplotlib font-cache inside the worktree so it never writes to
#    system TEMP (avoids Windows PermissionError).
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parent
_mpl_config = _REPO_ROOT / ".mpl_config"
_mpl_config.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config))

# ---------------------------------------------------------------------------
# 3. Redirect pytest tmp_path / tmp_path_factory basetemp into the worktree.
#    This avoids reliance on %TEMP% / AppData\Local\Temp where Windows
#    frequently raises PermissionError on cleanup.
#    NOTE: basetemp is a CLI-only option, so we set it here via the hook.
# ---------------------------------------------------------------------------
_LOCAL_BASETEMP = _REPO_ROOT / ".pytest_tmp"


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Override basetemp to a worktree-local directory when not explicitly set."""
    # Only override if the user didn't pass --basetemp on the CLI.
    if config.option.basetemp is None:
        config.option.basetemp = str(_LOCAL_BASETEMP)
