#!/usr/bin/env bash
#
# vivarium setup: a venv, three packages, and two fly brains.
#
# Safe to re-run. Downloads resume where they left off, and the matrix build is
# cached, so a second run costs a few seconds.
#
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

CHECK=0
NODATA=0
REBUILD=0

usage() {
  cat <<'USAGE'
usage: ./setup.sh [options]

  --check      report what is already in place, change nothing
  --no-data    set up the venv but skip the 1.5GB of connectome
  --rebuild    discard the cached matrices and rebuild them
  -h, --help   this

Needs: python 3.10+, ~2.1GB of disk, and a network connection for the first run.
USAGE
}

for a in "$@"; do
  case "$a" in
    --check)   CHECK=1 ;;
    --no-data) NODATA=1 ;;
    --rebuild) REBUILD=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $a" >&2; usage >&2; exit 2 ;;
  esac
done

say()  { printf '  %s\n' "$*"; }
warn() { printf '  !  %s\n' "$*" >&2; }
die()  { printf '\n  !! %s\n\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- python

py=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  command -v "$c" >/dev/null 2>&1 || continue
  v="$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)" || continue
  if [ "$(printf '%s\n3.10\n' "$v" | sort -V | head -1)" = "3.10" ]; then
    py="$c"; pyv="$v"; break
  fi
done
[ -n "$py" ] || die "need python 3.10 or newer; found none on PATH."

# ------------------------------------------------------------------ check

if [ "$CHECK" = 1 ]; then
  say "python     $py ($pyv)"
  if [ -x .venv/bin/python ]; then
    if .venv/bin/python -c 'import numpy, scipy, pyarrow' 2>/dev/null; then
      say "venv       ready"
    else
      say "venv       present but incomplete -- run ./setup.sh"
    fi
  else
    say "venv       MISSING -- run ./setup.sh"
  fi
  .venv/bin/python - <<'PY' 2>/dev/null || say "data       cannot check without the venv"
import sys; sys.path.insert(0, ".")
from vivarium import data
for k, name in data.LOCAL.items():
    p = data.data_dir() / name
    if p.exists():
        tag = " (symlinked)" if p.is_symlink() else ""
        print(f"  data       {name}  {p.stat().st_size/1e6:.0f} MB{tag}")
    else:
        part = data.partial(k)
        extra = f"  ({part/1e6:.0f} MB downloaded, will resume)" if part else ""
        print(f"  data       {name}  MISSING{extra}")
for sex in ("female", "male"):
    c = data.data_dir() / f"{sex}_brain_v3.npz"
    print(f"  matrix     {sex}: {'built' if c.exists() else 'not built yet'}")
PY
  exit 0
fi

# ------------------------------------------------------------------- disk

need_gb=2.1
if command -v df >/dev/null 2>&1; then
  free_kb="$(df -Pk "$here" | awk 'NR==2 {print $4}')"
  free_gb="$(awk -v k="$free_kb" 'BEGIN{printf "%.1f", k/1048576}')"
  if awk -v f="$free_gb" -v n="$need_gb" 'BEGIN{exit !(f < n)}'; then
    warn "only ${free_gb}GB free here; this wants about ${need_gb}GB."
    warn "continuing anyway -- downloads resume, so a disk-full is recoverable."
  else
    say "disk       ${free_gb}GB free"
  fi
fi

# ------------------------------------------------------------------- venv

if [ ! -x .venv/bin/python ]; then
  say "creating the venv ..."
  if ! "$py" -m venv .venv 2>/dev/null; then
    die "could not create a venv.
     On Debian/Ubuntu this usually means the venv module is missing:
         sudo apt install python3-venv"
  fi
fi
say "python     $py ($pyv)"

say "installing numpy, scipy, pyarrow ..."
.venv/bin/python -m pip install -q --upgrade pip >/dev/null 2>&1 || true
if ! .venv/bin/python -m pip install -q -r requirements.txt; then
  die "dependency install failed. If you are offline, that is why."
fi
.venv/bin/python -c 'import numpy, scipy, pyarrow' \
  || die "dependencies installed but will not import."
say "deps       ok"

# ------------------------------------------------------------------- data

if [ "$NODATA" = 1 ]; then
  say "data       skipped (--no-data); ./bin/jar will fetch on first run"
else
  say ""
  say "fetching the connectomes. About 1.5GB, from two public Google Cloud"
  say "buckets. No login, no API key. Interrupted downloads resume."
  say ""
  say "  female  FlyWire FAFB v783     CC BY-SA 4.0"
  say "  male    Janelia MaleCNS v1.0  CC BY 4.0"
  say ""
  .venv/bin/python - <<'PY' || die "download failed. Re-run ./setup.sh to resume."
import sys; sys.path.insert(0, ".")
from vivarium import data
for k in ("female_meta", "female_edges", "male_ann", "male_nt", "male_weights"):
    data.ensure(k)
PY
  say "data       ok"

  if [ "$REBUILD" = 1 ]; then
    rm -f data/female_brain_v*.npz data/male_brain_v*.npz
    say "discarded the cached matrices"
  fi

  say "building the connectivity matrices (one-time, under a minute) ..."
  .venv/bin/python - <<'PY' || die "matrix build failed."
import sys; sys.path.insert(0, ".")
from vivarium import connectome
for s in ("female", "male"):
    connectome.load(s)
PY
fi

cat <<'DONE'

  ready.

      ./bin/jar                 3 females, 3 males
      ./bin/jar -f 6 -m 6       more of them
      ./bin/jar -n 20           raise the simulated-brain limit
      ./bin/jar --census        what resolved in each connectome
      ./bin/jar --selftest      re-measure every claim the README makes
      ./bin/jar --bench 200     headless, report ticks/sec

  Needs a terminal at least 60x18.  Press ? inside for the keys.

DONE
