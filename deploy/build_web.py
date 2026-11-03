"""Build the static web version of the 3D simulator (runs entirely in the browser).

    python deploy/build_web.py [out_dir]        (default: dist/web3d)

Output: web3d/ front end + LICENSE and NOTICE + the simulator's Python sources (py/, listed in
py/manifest.json) + a trimmed
Pyodide distribution (pyodide/: runtime, numpy, scipy, clarabel and their
dependencies).  Serve the directory with any static web server; no Python
backend is needed.  Pyodide files are downloaded once into .cache/.

scipy is cut down to what the simulator and Clarabel import (see slim_scipy):
13.3 MB -> ~0.8 MB, and libopenblas (1.9 MB, needed only by scipy.linalg)
is no longer shipped.  Pass --full-scipy to ship the complete wheel.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYODIDE_VERSION = "0.28.3"
CDN = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/"
CORE = ["pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"]
PACKAGES = ["numpy", "scipy", "clarabel"]
SIM_PACKAGES = ["common", "lander3d"]
# scipy as shipped to the browser: the package's top-level modules, scipy._lib
# and scipy.sparse (the CSC matrices of common/conic.py, lander3d/boostback.py
# and Clarabel's Python API) -- nothing else is imported.  scipy.sparse 1.14
# imports csgraph eagerly (csgraph -> sparse.linalg -> scipy.linalg ->
# libopenblas); that import is the one line patched out.
SLIM_SCIPY_KEEP = re.compile(r"scipy/[^/]+\.py|scipy/_lib/.+|scipy/sparse/.+|scipy-[^/]+\.dist-info/.+")
SLIM_SCIPY_DROP = re.compile(r".*/tests/.*|scipy/_lib/cobyqa/.*|scipy/sparse/(csgraph|linalg)/.*")
SLIM_SCIPY_BUILD_TAG = "1sparse"     # wheel build tag: a new file name, so no browser cache mixes the two


def fetch(name: str, cache: Path) -> Path:
    path = cache / name
    if not path.exists():
        print(f"  download {name}", flush=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(CDN + name, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(path)
    return path


def _record_hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def slim_scipy(full: Path, target_dir: Path) -> Path:
    """Write the cut-down scipy wheel (see SLIM_SCIPY_KEEP) into ``target_dir``."""
    name, version, *tags = full.name[:-len(".whl")].split("-")
    target = target_dir / ("-".join([name, version, SLIM_SCIPY_BUILD_TAG, *tags[-3:]]) + ".whl")
    patch = "scipy/sparse/__init__.py"
    eager = "# For backward compatibility with v0.19.\nfrom . import csgraph\n"
    with zipfile.ZipFile(full) as src:
        files = {}
        for info in src.infolist():
            n = info.filename
            if info.is_dir() or not SLIM_SCIPY_KEEP.fullmatch(n) or SLIM_SCIPY_DROP.fullmatch(n):
                continue
            files[n] = src.read(n)
    text = files[patch].decode()
    if eager not in text:
        raise SystemExit(f"slim scipy: {patch} changed upstream; update SLIM_SCIPY_* or use --full-scipy")
    files[patch] = text.replace(eager, "# (csgraph and sparse.linalg are not shipped in this browser build)\n").encode()
    record = next(n for n in files if n.endswith(".dist-info/RECORD"))
    files[record] = "".join(f"{n},{_record_hash(d)},{len(d)}\n" for n, d in sorted(files.items()) if n != record).encode()
    files[record] += f"{record},,\n".encode()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as dst:
        for n in sorted(files):
            info = zipfile.ZipInfo(n, date_time=(2020, 1, 1, 0, 0, 0))   # reproducible: same bytes, same sha256
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            dst.writestr(info, files[n])
    return target


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    full_scipy = "--full-scipy" in sys.argv[1:]
    out = Path(args[0]) if args else ROOT / "dist" / "web3d"
    cache = ROOT / ".cache" / f"pyodide-{PYODIDE_VERSION}"
    cache.mkdir(parents=True, exist_ok=True)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "web3d", out)
    for name in ("LICENSE", "NOTICE"):          # Apache-2.0: redistributions carry both
        shutil.copy2(ROOT / name, out / name)
    sources = sorted(p.relative_to(ROOT).as_posix() for pkg in SIM_PACKAGES for p in (ROOT / pkg).rglob("*.py"))
    for rel in sources:
        target = out / "py" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    (out / "py" / "manifest.json").write_text(json.dumps(sources), encoding="utf-8")

    pyo = out / "pyodide"
    pyo.mkdir()
    for name in CORE:
        shutil.copy2(fetch(name, cache), pyo / name)
    lock_doc = json.loads((cache / "pyodide-lock.json").read_text(encoding="utf-8"))
    lock = lock_doc["packages"]
    if not full_scipy:
        slim = slim_scipy(fetch(lock["scipy"]["file_name"], cache), pyo)
        lock["scipy"]["file_name"] = slim.name
        lock["scipy"]["sha256"] = hashlib.sha256(slim.read_bytes()).hexdigest()
        lock["scipy"]["depends"] = [d for d in lock["scipy"]["depends"] if d != "libopenblas"]
        (pyo / "pyodide-lock.json").write_text(json.dumps(lock_doc), encoding="utf-8")
    need, todo = set(), list(PACKAGES)
    while todo:
        p = todo.pop()
        if p not in need:
            need.add(p)
            todo.extend(lock[p].get("depends", []))
    for p in sorted(need):
        if not (pyo / lock[p]["file_name"]).exists():
            shutil.copy2(fetch(lock[p]["file_name"], cache), pyo / lock[p]["file_name"])
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"built {out}  ({size / 1e6:.1f} MB, packages: {', '.join(sorted(need))})")


if __name__ == "__main__":
    main()
