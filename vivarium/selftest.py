"""Check the claims the README makes about the graphs.

Everything asserted in "Things that came out of the graph, not out of me" is a
measurement, and measurements rot. If a dataset is revised, or a selector stops
matching, this is what tells you -- rather than the jar quietly doing something
plausible and wrong.
"""
from __future__ import annotations

import sys

import numpy as np

from . import channels as CH
from . import connectome
from .engine import Cohort


def _probe(brain, channel: str, out: str, intensity: float, ticks: int = 300) -> float:
    co = Cohort(brain, capacity=1)
    s = co.claim()
    rng = np.random.default_rng(17)
    peak = 0.0
    for _ in range(ticks):
        co.clear_drive()
        co.stimulate(s, channel, intensity, rng)
        co.step()
        peak = max(peak, co.readout(s, out))
    return peak


def run() -> int:
    fails = []

    def check(ok: bool, what: str, detail: str = "") -> None:
        print(f"  {'ok  ' if ok else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
        if not ok:
            fails.append(what)

    print("loading both connectomes ...", file=sys.stderr)
    fem = connectome.load("female", quiet=True)
    mal = connectome.load("male", quiet=True)
    wf, wm = CH.resolve(fem), CH.resolve(mal)

    print("\nsizes")
    check(fem.n == 144_837, "female is 144,837 neurons", str(fem.n))
    check(mal.n == 166_700, "male is 166,700 neurons", str(mal.n))
    check(not fem.has_vnc and mal.has_vnc, "only the male has a nerve cord")

    print("\nchannels resolve")
    for name, w in (("female", wf), ("male", wm)):
        for key in ("light", "hearing", "pheromone", "co2", "taste", "touch"):
            check(key in w.inputs, f"{name}: {key} resolves",
                  f"{len(w.inputs.get(key, []))} cells")
        for key in ("escape", "backward", "descending", "courtship"):
            check(key in w.outputs, f"{name}: {key} resolves",
                  f"{len(w.outputs.get(key, []))} cells")

    print("\nthe asymmetries are real, not bugs")
    check("song" not in wf.outputs and "song" in wm.outputs,
          "pIP10/TN1 song: male only")
    check("wing" not in wf.outputs and "wing" in wm.outputs,
          "wing motor neurons: male only")
    check("legs" not in wf.outputs and "legs" in wm.outputs,
          "leg motor neurons: male only")
    check("oviposit" in wf.outputs and "oviposit" not in wm.outputs,
          "oviDN egg-laying: female only")
    check(len(wf.inputs["light"]) > 3 * len(wm.inputs["light"]),
          "she has far more photoreceptors than he does",
          f"{len(wf.inputs['light'])} vs {len(wm.inputs['light'])}")
    check(len(wm.inputs["taste"]) > 2 * len(wf.inputs["taste"]),
          "he has far more gustatory neurons (his dataset has legs)",
          f"{len(wm.inputs['taste'])} vs {len(wf.inputs['taste'])}")

    print("\nthe findings the README leans on")
    pher = _probe(fem, "pheromone", "courtship", 1.6)
    hear = _probe(fem, "hearing", "courtship", 1.6)
    touch = _probe(fem, "touch", "courtship", 1.6)
    check(pher > 0.0, "her pC1 responds to cVA", f"peak {pher:.4f}")
    check(hear == 0.0, "her pC1 does NOT respond to song/hearing", f"peak {hear:.4f}")
    check(touch == 0.0, "her pC1 does NOT respond to touch", f"peak {touch:.4f}")

    ovi = max(_probe(fem, ch, "oviposit", 1.6) for ch in
              ("taste", "touch", "pheromone", "food_smell", "light"))
    check(ovi == 0.0, "oviDN never fires from any stimulus", f"peak {ovi:.4f}")

    smell = _probe(fem, "food_smell", "descending", 1.6)
    check(smell < 0.005, "smelling food barely reaches a descending neuron",
          f"peak {smell:.5f}")

    taste = _probe(fem, "taste", "proboscis", 1.6)
    check(taste > 0.0, "tasting food does reach the proboscis motor neurons",
          f"peak {taste:.4f}")

    # Her DNa01/DNa02 answer the photoreceptors and nothing else we can
    # deliver; his answer nothing at all. Probing with touch, as an earlier
    # version of this test did, drives neither and proves nothing.
    steer_f = _probe(fem, "light", "steer", 1.6)
    steer_m = max(_probe(mal, ch, "steer", 1.6)
                  for ch in ("light", "touch", "hearing", "jolt", "wind"))
    check(steer_f > 0.0, "her DNa01/DNa02 steering answers light",
          f"peak {steer_f:.3f}")
    check(steer_m == 0.0, "his DNa01/DNa02 answer nothing we can deliver",
          f"peak {steer_m:.3f}")

    print()
    if fails:
        print(f"{len(fails)} check(s) FAILED:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("all checks passed -- the README is still telling the truth.")
    return 0
