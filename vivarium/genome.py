"""What gets inherited.

We do not mutate the connectome itself. Rewiring 25 million edges per fly would
cost 100MB a head and, worse, it would be fiction -- nobody has measured what a
rewired fly's graph looks like. Instead a genome is a short list of *gains*:
how hard each sensory channel drives, how excitable the brain is overall, how
fast the body burns energy. The wiring underneath stays exactly as it was
measured. Only the volume knobs are heritable.

This is closer to the truth than rewiring would be. Most real behavioural
variation between flies is receptor expression and gain, not a different graph.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

# gene -> (default, low, high, blurb)
GENES = {
    "exc":        (1.00, 0.55, 1.65, "gain on every excitatory (cholinergic) synapse"),
    "inh":        (1.00, 0.40, 1.80, "gain on every inhibitory (GABA/glutamate) synapse"),
    "thr":        (0.12, 0.07, 0.22, "membrane threshold -- low is a twitchy fly"),
    "leak":       (0.94, 0.86, 0.975, "charge retained per tick"),
    "hearing":    (1.00, 0.20, 2.40, "drive into Johnston's organ"),
    "smell":      (1.00, 0.20, 2.40, "drive into the olfactory receptor neurons"),
    "pheromone":  (1.00, 0.10, 3.00, "drive into ORN_DA1 specifically -- the cVA line"),
    "taste":      (1.00, 0.20, 2.40, "drive into the gustatory neurons"),
    "sight":      (1.00, 0.10, 2.20, "drive into the photoreceptors"),
    "touch":      (1.00, 0.20, 2.40, "drive into the bristles"),
    "metabolism": (1.00, 0.60, 1.60, "how fast she spends herself"),
    "nerve":      (1.00, 0.25, 2.50, "how much the giant fibre has to say before she goes"),
    "vigour":     (1.00, 0.45, 1.70, "descending command -> actual movement"),
    "lust":       (1.00, 0.10, 2.80, "gain on the courtship readout"),
}

DEFAULTS = {k: v[0] for k, v in GENES.items()}
BOUNDS = {k: (v[1], v[2]) for k, v in GENES.items()}
BLURB = {k: v[3] for k, v in GENES.items()}

# Named for the fact that a fly in a jar is a specimen before it is anything
# else. First half is the accession number, second half is what you end up
# calling her by about ten minutes in.
_SYLL_A = ("mor", "vel", "hes", "dro", "ang", "pal", "cir", "nyx", "vor", "tes",
           "gla", "sem", "hol", "mar", "pyx", "cal", "ver", "ost", "lin", "fen")
_SYLL_B = ("ith", "andra", "ise", "ope", "usa", "illa", "ex", "oma", "ara", "yne",
           "ade", "icia", "ora", "ella", "is", "yx", "una", "eth", "ira", "ola")


def coin_name(rng: random.Random, sex: str) -> str:
    return (rng.choice(_SYLL_A) + rng.choice(_SYLL_B)).capitalize()


@dataclass
class Genome:
    genes: dict = field(default_factory=lambda: dict(DEFAULTS))
    generation: int = 0
    parents: tuple = ()

    def __getitem__(self, k: str) -> float:
        return self.genes.get(k, DEFAULTS[k])

    @staticmethod
    def wild(rng: random.Random, spread: float = 0.12) -> "Genome":
        """A founder. Not identical to the textbook fly, but near it."""
        g = {}
        for k, d in DEFAULTS.items():
            lo, hi = BOUNDS[k]
            g[k] = _clamp(d * rng.lognormvariate(0.0, spread), lo, hi)
        return Genome(genes=g, generation=0)

    def mutated(self, rng: random.Random, rate: float = 0.25,
                strength: float = 0.14) -> "Genome":
        g = dict(self.genes)
        for k in g:
            if rng.random() < rate:
                lo, hi = BOUNDS[k]
                g[k] = _clamp(g[k] * rng.lognormvariate(0.0, strength), lo, hi)
        return replace(self, genes=g)

    def divergence(self, other: "Genome") -> float:
        """How different two flies are, 0 = clones. Used for the pedigree panel."""
        acc = 0.0
        for k in DEFAULTS:
            lo, hi = BOUNDS[k]
            acc += abs(self[k] - other[k]) / (hi - lo)
        return acc / len(DEFAULTS)

    def notable(self, n: int = 3) -> list:
        """The genes furthest from wild type -- what makes this one itself."""
        scored = []
        for k, d in DEFAULTS.items():
            lo, hi = BOUNDS[k]
            scored.append((abs(self[k] - d) / (hi - lo), k, self[k], d))
        scored.sort(reverse=True)
        return [(k, v, d) for _, k, v, d in scored[:n]]


def cross(a: Genome, b: Genome, rng: random.Random) -> Genome:
    """Meiosis, roughly. Each gene comes from one parent or blends the two."""
    g = {}
    for k in DEFAULTS:
        lo, hi = BOUNDS[k]
        r = rng.random()
        if r < 0.42:
            v = a[k]
        elif r < 0.84:
            v = b[k]
        else:                      # a recombinant somewhere between them
            t = rng.random()
            v = a[k] * t + b[k] * (1 - t)
        g[k] = _clamp(v, lo, hi)
    return Genome(genes=g, generation=max(a.generation, b.generation) + 1)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v
