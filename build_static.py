"""Pre-render the AFB browser into a static `dist/` site for GitHub Pages.

Usage:
    cd main/mcq_eval/AFB_webpage
    URL_PREFIX=/afb STATIC_MODE=1 python build_static.py
    # output: ./dist/

Then publish dist/ to a `gh-pages` branch (or use the bundled GitHub Action).

Env:
    URL_PREFIX  optional path subprefix (e.g. "/afb" for github.com/<user>/afb)
    STATIC_MODE "1" hides server-only buttons (reshuffle, load-20)
    STATIC_SEED int seed for the per-cell random sample (default: deterministic)
    STATIC_N    int number of questions per cell (default: 20)
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

# Make sure STATIC_MODE / URL_PREFIX env vars are visible BEFORE importing app.
os.environ.setdefault("STATIC_MODE", "1")

import app as flask_app  # noqa: E402

DIST = Path(__file__).resolve().parent / "dist"
SEED = int(os.environ.get("STATIC_SEED", "42"))
N    = int(os.environ.get("STATIC_N",    "20"))


def _write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


def main() -> int:
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)

    client = flask_app.app.test_client()

    # 1. Index
    print("[1/3] rendering /")
    r = client.get("/")
    assert r.status_code == 200, f"index failed: {r.status_code}"
    _write(DIST / "index.html", r.data)

    # 2. .nojekyll so GitHub Pages serves _underscore-prefixed paths if any
    _write(DIST / ".nojekyll", b"")

    # 3. Iterate every cell
    rows = flask_app._load_afb_rows()
    fms  = flask_app.FAILURE_MODES
    cells = []
    for r in rows:
        target = r["target_model"]
        ds_url = "baseline" if r["ft_dataset"] == "N/A" else r["ft_dataset"]
        for fm in fms:
            cells.append((target, ds_url, fm))
    print(f"[2/3] rendering {len(cells)} cell pages…")

    t0 = time.time()
    failed = 0
    for i, (target, ds_url, fm) in enumerate(cells, 1):
        url = f"/cell/{target}/{ds_url}/{fm}/?n={N}&seed={SEED}"
        r = client.get(url)
        if r.status_code != 200:
            failed += 1
            print(f"  ! {url} → HTTP {r.status_code}")
            continue
        out = DIST / "cell" / target / ds_url / fm / "index.html"
        _write(out, r.data)
        if i % 200 == 0 or i == len(cells):
            elapsed = time.time() - t0
            rate    = i / elapsed if elapsed > 0 else 0
            eta     = (len(cells) - i) / rate if rate > 0 else 0
            print(f"  {i}/{len(cells)}  ({rate:.0f}/s, eta {eta:.0f}s)")

    print(f"[3/3] done. failed: {failed}/{len(cells)}")
    print(f"output: {DIST}")
    print()
    print("Total size:")
    os.system(f"du -sh '{DIST}'")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
