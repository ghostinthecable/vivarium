"""Fetch the two connectomes this jar needs.

Both are public, both are free, neither wants a login.

  female   FlyWire FAFB v783        144,837 neurons, brain only.       CC BY-SA 4.0
  male     Janelia MaleCNS v1.0     211,577 segments, brain + nerve cord. CC BY 4.0

The asymmetry is real and it matters: the female dataset stops at the neck, so
she has descending commands and no body to send them to. The male dataset runs
all the way down the nerve cord to the muscles. We do not paper over this.
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

FAFB = ("https://storage.googleapis.com/"
        "lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/fafb_783")
MCNS = ("https://storage.googleapis.com/flyem-male-cns/v1.0/"
        "connectome-data/flat-connectome")

# local name -> (url, approx bytes)
FILES = {
    "female_meta":  (f"{FAFB}/fafb_783_meta.feather",                              13_539_866),
    "female_edges": (f"{FAFB}/fafb_783_simple_edgelist.feather",                  302_625_658),
    "male_ann":     (f"{MCNS}/body-annotations-male-cns-v1.0-minconf-0.5.feather",  14_483_314),
    "male_nt":      (f"{MCNS}/body-neurotransmitters-male-cns-v1.0.feather",        43_282_834),
    "male_weights": (f"{MCNS}/connectome-weights-male-cns-v1.0-minconf-0.5.feather", 1_100_000_000),
}

LOCAL = {
    "female_meta":  "fafb_783_meta.feather",
    "female_edges": "fafb_783_simple_edgelist.feather",
    "male_ann":     "male_annotations.feather",
    "male_nt":      "male_neurotransmitters.feather",
    "male_weights": "male_weights.feather",
}

# If you already cloned flyclaude, its 316MB of female fly is reused rather
# than downloaded twice. Set VIVARIUM_FLYBRAIN to point somewhere else.
SIBLING = Path(os.environ.get("VIVARIUM_FLYBRAIN", "/opt/flybrain/data"))


def data_dir() -> Path:
    d = Path(os.environ.get("VIVARIUM_DATA",
                            Path(__file__).resolve().parent.parent / "data"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _borrow(key: str, dest: Path) -> bool:
    """Symlink a file we already have from a neighbouring flyclaude checkout."""
    src = SIBLING / LOCAL[key]
    if src.exists() and not dest.exists():
        dest.symlink_to(src)
        return True
    return False


def _remote_size(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


def _download(url: str, dest: Path, expected: int) -> None:
    """Fetch with resume. The male edge list is 1.1GB and connections drop."""
    total = _remote_size(url) or expected
    tmp = dest.with_suffix(dest.suffix + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0

    if have and have == total:
        tmp.replace(dest)
        return
    if have > total:                    # stale or corrupt partial; start over
        tmp.unlink()
        have = 0

    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
        print(f"  resuming {dest.name} at {have / 1e6:.0f} MB "
              f"of {total / 1e6:.0f} MB ...", file=sys.stderr)
    else:
        print(f"  fetching {dest.name} ({total / 1e6:.0f} MB) ...", file=sys.stderr)

    with urllib.request.urlopen(req, timeout=60) as r:
        # A server that ignores Range replies 200 and sends the whole file.
        mode = "ab" if (have and r.status == 206) else "wb"
        if mode == "wb":
            have = 0
        got = have
        with open(tmp, mode) as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                got += len(chunk)
                pct = 100 * got / total if total else 0
                print(f"\r    {got / 1e6:7.1f} MB  {min(pct, 100.0):5.1f}%",
                      end="", file=sys.stderr)
    print(file=sys.stderr)

    final = tmp.stat().st_size
    if total and final != total:
        raise SystemExit(
            f"\n{dest.name}: got {final:,} bytes, expected {total:,}.\n"
            f"  The partial file is kept at {tmp} -- run setup.sh again to resume."
        )
    tmp.replace(dest)


def ensure(key: str) -> Path:
    dest = data_dir() / LOCAL[key]
    if dest.exists():
        return dest
    if _borrow(key, dest):
        print(f"  reusing {dest.name} from {SIBLING}", file=sys.stderr)
        return dest
    url, size = FILES[key]
    _download(url, dest, size)
    return dest


def ensure_sex(sex: str) -> dict[str, Path]:
    keys = ("female_meta", "female_edges") if sex == "female" else \
           ("male_ann", "male_nt", "male_weights")
    return {k: ensure(k) for k in keys}


def have(key: str) -> bool:
    return (data_dir() / LOCAL[key]).exists()


def partial(key: str) -> int:
    """Bytes already downloaded for an interrupted fetch, 0 if none."""
    p = data_dir() / (LOCAL[key] + ".part")
    return p.stat().st_size if p.exists() else 0


def total_bytes() -> int:
    return sum(sz for _url, sz in FILES.values())
