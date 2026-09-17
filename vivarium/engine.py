"""Stepping a dozen whole connectomes at once, fast enough to watch.

The trick is that every fly of a given sex shares one connectivity matrix --
the wiring is the same measured wiring, that is the whole point -- so what
differs between flies is only their *state* and their *gains*. So we keep
state as one dense (neurons x flies) array and do a single sparse-times-dense
product per tick. Twelve flies cost barely more than one.

Genome gains ride on that product as per-fly scalars:

    input = (Wexc @ S) * exc[fly] + (Winh @ S) * inh[fly] + sensory

which is why excitation and inhibition are stored as two separate matrices.
It also means a GABA blocker is one number, and so is a nervous fly.

Dynamics are leaky integrate-and-fire: charge accumulates, neuron pops, charge
resets, brief refractory period. Real neurons have dendrites, synaptic delays,
neuropeptides and opinions. These have a float and a threshold. The wiring is
measured; the dynamics are a cartoon, and every project in this space makes the
same trade.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import channels as CH

# csr_matvecs is single-threaded and this machine is not. Splitting the matrix
# into row blocks and running them on a small pool is a clean ~2.5x: the blocks
# write to disjoint output rows, so there is nothing to synchronise, and scipy
# drops the GIL for the duration. More than four threads loses to memory
# bandwidth -- measured, not assumed.
THREADS = max(1, min(4, (os.cpu_count() or 1)))

# How fast a sense organ stops reporting a constant. 0.02 is roughly a
# two-second time constant at ten ticks a second. Not every sense adapts at the
# same speed, and the differences matter here: photoreceptors adapt fast (they
# have to -- 19,250 of them under a steady light will otherwise drown
# everything else), while the pheromone and CO2 lines are tonic signals that a
# fly is supposed to keep reporting. Adapting cVA at the default rate pushed it
# below the threshold that makes pC1 fire, and mating stopped happening at all.
ADAPT_RATE = 0.02
ADAPT_RATES = {
    "light": 0.035,
    "pheromone": 0.004,
    "co2": 0.0025,
    "heat": 0.006,
    "damp": 0.006,
}
ADAPT_FLOOR = 0.25


class Blocks:
    """Row-partitioned copies of the two signed matrices, applied in parallel.

    Computes  Wexc @ Se + Winh @ Si  in one pass. The per-fly excitation and
    inhibition gains are folded into Se and Si by the caller rather than
    applied to the result, which is algebraically identical -- they are
    per-column scalars, and column scaling commutes with left multiplication --
    and lets both matrices share a single traversal.
    """

    def __init__(self, Wexc, Winh, threads: int = THREADS):
        n = Wexc.shape[0]
        self.n = n
        self.threads = threads
        self.bounds = [n * i // threads for i in range(threads + 1)]
        self.exc = [Wexc[self.bounds[i]:self.bounds[i + 1]] for i in range(threads)]
        self.inh = [Winh[self.bounds[i]:self.bounds[i + 1]] for i in range(threads)]
        self.pool = ThreadPoolExecutor(max_workers=threads,
                                       thread_name_prefix="matvec") if threads > 1 else None

    def combine(self, Se, Si, out) -> None:
        if self.pool is None:
            out[:] = self.exc[0].dot(Se) + self.inh[0].dot(Si)
            return

        def part(i):
            a, b = self.bounds[i], self.bounds[i + 1]
            out[a:b] = self.exc[i].dot(Se)
            out[a:b] += self.inh[i].dot(Si)

        list(self.pool.map(part, range(self.threads)))


class Cohort:
    """All the live adults of one sex, stepped together."""

    def __init__(self, brain, capacity: int = 24):
        self.brain = brain
        self.wiring = CH.resolve(brain)
        self.blocks = Blocks(brain.Wexc, brain.Winh)
        self.n = brain.n
        self.capacity = capacity

        z = np.zeros((self.n, capacity), dtype=np.float32)
        self.V = z
        self.S = np.zeros((self.n, capacity), dtype=np.float32)
        self.R = np.zeros((self.n, capacity), dtype=np.int8)
        self.drive = np.zeros((self.n, capacity), dtype=np.float32)

        # per-fly parameters, filled from genomes each time the roster changes
        self.exc = np.ones(capacity, dtype=np.float32)
        self.inh = np.ones(capacity, dtype=np.float32)
        self.thr = np.full(capacity, 0.12, dtype=np.float32)
        self.leak = np.full(capacity, 0.94, dtype=np.float32)
        self.live = np.zeros(capacity, dtype=bool)

        self.free = list(range(capacity))
        self.refractory = 2

        # rolling spike totals per output channel, decayed -- this is what the
        # body actually reads, so a single tick's noise doesn't throw a fly
        self.out = {k: np.zeros(capacity, dtype=np.float32) for k in self.wiring.outputs}
        self.spiked = np.zeros(capacity, dtype=np.int32)

        # Sensory adaptation, per channel per fly. Every real receptor does
        # this: hold a smell or a light steady and the firing falls away within
        # seconds, which is why a fly in a lit jar is not permanently blinded
        # and permanently escaping. Without it the photoreceptors alone pin the
        # Giant Fibre at maximum forever, which is what this model did at first.
        self.adapt = {k: np.zeros(capacity, dtype=np.float32)
                      for k in self.wiring.inputs}

    # ------------------------------------------------------------- roster

    def claim(self) -> int:
        if not self.free:
            return -1
        s = self.free.pop(0)
        self.V[:, s] = 0.0
        self.S[:, s] = 0.0
        self.R[:, s] = 0
        self.drive[:, s] = 0.0
        self.live[s] = True
        for v in self.out.values():
            v[s] = 0.0
        for v in self.adapt.values():
            v[s] = 0.0
        return s

    def release(self, slot: int) -> None:
        if slot < 0:
            return
        self.live[slot] = False
        self.V[:, slot] = 0.0
        self.S[:, slot] = 0.0
        self.drive[:, slot] = 0.0
        for v in self.out.values():
            v[slot] = 0.0
        if slot not in self.free:
            self.free.append(slot)

    def configure(self, slot: int, genome) -> None:
        self.exc[slot] = genome["exc"]
        self.inh[slot] = genome["inh"]
        self.thr[slot] = genome["thr"]
        self.leak[slot] = genome["leak"]

    # ------------------------------------------------------------ stimulus

    def clear_drive(self) -> None:
        self.drive[:, :] = 0.0

    def stimulate(self, slot: int, channel: str, amount: float,
                  rng: np.random.Generator | None = None) -> int:
        """Inject current into the real receptor neurons for `channel`.

        Intensity is graded two ways at once, because that is how a sense organ
        actually works: a stronger smell both recruits *more* receptor neurons
        and drives each of them *harder*. Doing only the second does nothing at
        all -- an integrate-and-fire neuron that is over threshold fires at the
        same rate whether you give it 1.2 or 8.0, because the spike throws the
        surplus away. Recruitment is what carries intensity.
        """
        idx = self.wiring.inputs.get(channel)
        if idx is None or amount <= 0.0:
            return 0
        a = self.adapt[channel]
        a[slot] += (float(amount) - a[slot]) * ADAPT_RATES.get(channel, ADAPT_RATE)
        # Respond to the part of the signal that is *news*. A steady background
        # still gets through at ADAPT_FLOOR, so a bright jar is never quite the
        # same as a dark one.
        effective = max(float(amount) - a[slot] * (1.0 - ADAPT_FLOOR), 0.0)
        if effective <= 1e-4:
            return 0
        frac = 1.0 - np.exp(-effective)                  # saturating recruitment
        amp = self.thr[slot] * (0.55 + 0.95 * min(effective, 2.0))
        if rng is not None and frac < 0.999:
            take = idx[rng.random(len(idx)) < frac]
        else:
            take = idx
        if len(take) == 0:
            return 0
        self.drive[take, slot] += amp
        return len(take)

    # ---------------------------------------------------------------- step

    @property
    def k(self) -> int:
        """How many state columns are actually occupied.

        The sparse product costs time proportional to this, not to capacity,
        so a half-empty jar really is half the work.
        """
        w = np.flatnonzero(self.live)
        return int(w[-1]) + 1 if len(w) else 0

    def step(self, gaba_block: float = 1.0, noise: float = 0.0,
             rng: np.random.Generator | None = None) -> None:
        k = self.k
        if k == 0:
            return
        S = self.S[:, :k]

        # Fold each fly's gains into the spike vector, not the product.
        Se = S * self.exc[None, :k]
        Si = S * (self.inh[None, :k] * gaba_block)
        inp = self._scratch(k)
        self.blocks.combine(Se, Si, inp)
        inp += self.drive[:, :k]

        if noise > 0.0 and rng is not None:
            inp += rng.standard_normal(inp.shape, dtype=np.float32) * noise

        V = self.V[:, :k]
        V *= self.leak[None, :k]
        V += inp
        np.maximum(V, 0.0, out=V)
        R = self.R[:, :k]
        V *= (R == 0)                       # refractory cells hold at zero

        fired = V >= self.thr[None, :k]
        fired &= self.live[None, :k]
        self.S[:, :k] = fired

        np.subtract(R, 1, out=R, where=R > 0)
        R[fired] = self.refractory
        # Reset by subtraction, not to zero: charge above threshold carries
        # into the next tick instead of being thrown away, so how hard a neuron
        # is driven survives the spike and its rate means something. Written as
        # a broadcast multiply because the masked form allocated a whole
        # (neurons x flies) array every tick.
        V -= self.S[:, :k] * self.thr[None, :k]
        np.maximum(V, 0.0, out=V)

        self.spiked[:] = 0
        self.spiked[:k] = fired.sum(axis=0).astype(np.int32)

        # Decayed readout per output population -- a rate, not a single tick.
        for key, idx in self.wiring.outputs.items():
            hit = self.S[idx, :k].sum(axis=0) / max(len(idx), 1)
            self.out[key] *= 0.82
            self.out[key][:k] += hit

    def _scratch(self, k: int) -> np.ndarray:
        buf = getattr(self, "_buf", None)
        if buf is None or buf.shape[1] != k:
            buf = np.zeros((self.n, k), dtype=np.float32)
            self._buf = buf
        return buf

    def readout(self, slot: int, key: str) -> float:
        v = self.out.get(key)
        return 0.0 if v is None else float(v[slot])

    def top_descending(self, slot: int, k: int = 6) -> list:
        """Which named command neurons are actually firing right now."""
        idx = self.wiring.outputs.get("descending")
        if idx is None:
            return []
        s = self.S[idx, slot]
        hot = np.flatnonzero(s > 0)
        if len(hot) == 0:
            return []
        names = self.brain.celltype[idx[hot]]
        out, seen = [], {}
        for nm in names:
            nm = nm or "unnamed"
            seen[nm] = seen.get(nm, 0) + 1
        for nm, c in sorted(seen.items(), key=lambda kv: -kv[1])[:k]:
            out.append((nm, c))
        return out
