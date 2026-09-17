"""Command line. Mostly this just decides how many flies and gets out of the way."""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="jar",
        description="A sealed jar with real fly connectomes in it.")
    p.add_argument("-f", "--females", type=int, default=3,
                   help="female founders (FlyWire FAFB v783, brain only)")
    p.add_argument("-m", "--males", type=int, default=3,
                   help="male founders (Janelia MaleCNS v1.0, brain + nerve cord)")
    p.add_argument("-n", "--capacity", type=int, default=12,
                   help="how many brains can be simulated at once, per sex")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--food", type=int, default=1, help="sugar drops to start with")
    p.add_argument("--dark", action="store_true", help="start with the light off")
    p.add_argument("--selftest", action="store_true",
                   help="verify every factual claim the README makes about the graphs")
    p.add_argument("--census", action="store_true",
                   help="print what resolved in each connectome and exit")
    p.add_argument("--bench", type=int, metavar="TICKS",
                   help="run headless for TICKS and report the rate")
    a = p.parse_args(argv)

    if a.selftest:
        from .selftest import run
        return run()

    if a.census:
        return census()

    from .sim import Vivarium
    print("  loading two connectomes ...", file=sys.stderr)
    viv = Vivarium(capacity=a.capacity, seed=a.seed)
    viv.populate(a.females, a.males)
    for _ in range(a.food):
        viv.jar.feed()
    if a.dark:
        viv.jar.toggle_light()

    if a.bench:
        import time
        for _ in range(10):
            viv.tick()
        t0 = time.time()
        for _ in range(a.bench):
            viv.tick()
        dt = time.time() - t0
        n = len(viv.jar.adults())
        print(f"{a.bench} ticks, {n} flies: {dt:.1f}s -> {a.bench / dt:.1f} ticks/s")
        return 0

    from . import tui
    tui.run(viv)
    return 0


def census() -> int:
    from . import channels as CH
    from . import connectome
    for sex in ("female", "male"):
        b = connectome.load(sex)
        w = CH.resolve(b)
        print(f"\n{sex.upper()}  {b.n:,} neurons  "
              f"{b.Wexc.nnz + b.Winh.nnz:,} signed edges  "
              f"{'brain + nerve cord' if b.has_vnc else 'brain only'}")
        for d, k, label, n in CH.census(b, w):
            print(f"  {d:<3s} {k:<11s} {n:>7,d}  {label}")
        for k, label, why in w.absent:
            print(f"  --  {k:<11s} {'':>7s}  ABSENT: {why}")
    return 0
