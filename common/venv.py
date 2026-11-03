"""Re-run the entry point with the project's .venv interpreter when dependencies are missing.

Standard library only: this runs before numpy & co. can be imported.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REQUIRED = ("numpy", "scipy", "clarabel")
_GUARD = "GFOLD_VENV_REEXEC"


def _venv_python() -> Path | None:
    root = Path(__file__).resolve().parent.parent
    for rel in (".venv/bin/python", ".venv/Scripts/python.exe", "venv/bin/python", "venv/Scripts/python.exe"):
        p = root / rel
        if p.exists():
            return p
    return None


def ensure_venv() -> None:
    """If a required module is missing, restart this script under the project venv (once)."""
    if all(importlib.util.find_spec(m) is not None for m in REQUIRED):
        return
    py = _venv_python()
    if py is None or os.environ.get(_GUARD) or Path(sys.prefix).resolve() == py.parent.parent.resolve():
        missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
        sys.exit(
            f"缺少依赖: {', '.join(missing)}。请先创建虚拟环境并安装依赖:\n"
            "  python -m venv .venv && .venv/bin/pip install -r requirements.txt"
        )
    env = dict(os.environ, **{_GUARD: "1"})
    args = [str(py), *sys.argv]
    if os.name == "nt":
        sys.exit(subprocess.call(args, env=env))
    os.execve(str(py), args, env)
