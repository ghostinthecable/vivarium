"""One fly: a body, a genome, and a slot in the batched brain.

The brain state itself does not live here -- it lives as a column in a big
(neurons x flies) matrix over in engine.py, because that is the only way to
step twelve whole connectomes fast enough to watch. This object holds the fly's
body, and the index of its column.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .genome import Genome, coin_name

# Side-on jar. A fly is either on a surface or in the air. Walking flies do not
# buzz; flying flies do, and that is what everyone else hears.
FLOOR, LEFT, RIGHT, CEILING, AIR = "floor", "left", "right", "ceiling", "air"

EGG, LARVA, PUPA, ADULT, DEAD = "egg", "larva", "pupa", "adult", "dead"

# Ticks at the default rate. A fly gets about twenty minutes of your attention
# and then dies of it.
LIFESPAN = 9000
EGG_TICKS = 260
LARVA_TICKS = 900
PUPA_TICKS = 700


@dataclass
class Fly:
    ident: int
    name: str
    sex: str                       # "female" | "male"
    genome: Genome

    x: float = 0.0
    y: float = 0.0
    surface: str = FLOOR
    heading: float = 1.0           # +1 / -1 along the surface
    vx: float = 0.0
    vy: float = 0.0

    stage: str = ADULT
    age: int = 0
    energy: float = 1.0
    slot: int = -1                 # column in the brain state matrix, -1 = not simulated

    # what the brain most recently told the body to do
    drive: float = 0.0             # generic descending output
    buzz: float = 0.0              # wing amplitude -> what others hear
    song: float = 0.0              # male courtship song -> what she hears
    escape: float = 0.0
    backward: float = 0.0
    freeze: float = 0.0
    proboscis: float = 0.0
    oviposit: float = 0.0
    receptive: float = 0.0         # pC1 activity, female

    # state that is about other flies
    mated: bool = False
    gravid: float = 0.0
    mate_genome: object = None
    courting: int = -1             # ident of who she is being sung at by
    courted: float = 0.0           # male: decaying 'has been singing lately'
    last_spikes: int = 0
    takeoff_lock: int = 0
    dead_at: int = -1
    cause: str = ""

    history: list = field(default_factory=list)

    # ------------------------------------------------------------------ body

    @property
    def alive(self) -> bool:
        return self.stage != DEAD

    @property
    def simulated(self) -> bool:
        """Only adults have a connectome in this jar. Larvae are a different
        animal with a different (3,016-neuron) wiring diagram we do not load."""
        return self.stage == ADULT and self.slot >= 0

    @property
    def glyph(self) -> str:
        if self.stage == DEAD:
            return "x"
        if self.stage == EGG:
            return "."
        if self.stage == LARVA:
            return "~"
        if self.stage == PUPA:
            return "o"
        if self.surface == AIR:
            return "*"
        return "%" if self.sex == "male" else "#"

    def die(self, tick: int, cause: str) -> None:
        self.stage = DEAD
        self.dead_at = tick
        self.cause = cause
        self.vx = self.vy = 0.0
        self.buzz = self.song = 0.0

    def note(self, tick: int, what: str) -> None:
        self.history.append((tick, what))
        if len(self.history) > 40:
            del self.history[0]


def founder(ident: int, sex: str, rng: random.Random, w: float, h: float) -> Fly:
    f = Fly(
        ident=ident,
        name=coin_name(rng, sex),
        sex=sex,
        genome=Genome.wild(rng),
        x=rng.uniform(w * 0.15, w * 0.85),
        y=0.0,
        surface=FLOOR,
        heading=rng.choice((-1.0, 1.0)),
        energy=rng.uniform(0.75, 1.0),
        age=rng.randint(0, 400),
    )
    return f


def hatchling(ident: int, mother: Fly, father_genome: Genome,
              rng: random.Random) -> Fly:
    from .genome import cross
    g = cross(mother.genome, father_genome, rng).mutated(rng)
    sex = "male" if rng.random() < 0.5 else "female"
    return Fly(
        ident=ident,
        name=coin_name(rng, sex),
        sex=sex,
        genome=g,
        x=mother.x,
        y=mother.y,
        surface=FLOOR,
        stage=EGG,
        age=0,
        energy=0.62,
    )


# ------------------------------------------------------------------- physics

def step_body(f: Fly, w: float, h: float, gravity: float, rng: random.Random) -> None:
    """Move one fly one tick. Commands have already been read out of its brain."""
    if f.stage != ADULT:
        return

    vig = f.genome["vigour"]

    if f.surface == AIR:
        # Wing thrust is the buzz. No buzz, no lift, and the jar has a floor.
        lift = f.buzz * 0.85 * vig
        f.vy += lift - gravity
        f.vx += (f.heading * f.drive * 0.35 * vig
                 + rng.uniform(-0.09, 0.09))
        f.vx *= 0.92
        f.vy *= 0.92
        f.x += f.vx
        f.y += f.vy
        _land(f, w, h)
    else:
        speed = f.drive * 0.55 * vig
        if f.freeze > 0.35:
            speed = 0.0
        if f.backward > 0.25:
            speed = -speed * 0.8           # MDN. She really does walk backwards.
        along = speed * f.heading
        _walk(f, along, w, h)

        # The Giant Fibre does not ask. It launches. But it does not launch
        # twice in a row: the real escape circuit has a long refractory period
        # and habituates hard to repeated looming, which is why you eventually
        # get the fly with the glass.
        if f.takeoff_lock > 0:
            f.takeoff_lock -= 1
        elif f.escape > 0.5 * f.genome["nerve"]:
            _launch(f, rng)
            f.takeoff_lock = 55


def _walk(f: Fly, along: float, w: float, h: float) -> None:
    """Move along the current surface, turning the corner when it runs out."""
    if f.surface == FLOOR:
        f.x += along
        f.y = 0.0
        if f.x < 0.0:
            f.x, f.surface, f.y = 0.0, LEFT, 0.0
        elif f.x > w:
            f.x, f.surface, f.y = w, RIGHT, 0.0
    elif f.surface == LEFT:
        f.y += along
        f.x = 0.0
        if f.y < 0.0:
            f.y, f.surface = 0.0, FLOOR
        elif f.y > h:
            f.y, f.surface = h, CEILING
    elif f.surface == RIGHT:
        f.y += along
        f.x = w
        if f.y < 0.0:
            f.y, f.surface = 0.0, FLOOR
        elif f.y > h:
            f.y, f.surface = h, CEILING
    elif f.surface == CEILING:
        f.x -= along
        f.y = h
        if f.x < 0.0:
            f.x, f.surface = 0.0, LEFT
        elif f.x > w:
            f.x, f.surface = w, RIGHT


def _launch(f: Fly, rng: random.Random) -> None:
    """Jump off whatever she is standing on -- away from it, not into it."""
    surf = f.surface
    f.surface = AIR
    if surf == CEILING:
        f.vy = -rng.uniform(1.6, 3.0)
        f.vx = rng.uniform(-1.4, 1.4)
        f.y -= 0.6
    elif surf == LEFT:
        f.vx = rng.uniform(1.6, 3.0)
        f.vy = rng.uniform(0.2, 1.4)
        f.x += 0.6
    elif surf == RIGHT:
        f.vx = -rng.uniform(1.6, 3.0)
        f.vy = rng.uniform(0.2, 1.4)
        f.x -= 0.6
    else:
        f.vx = rng.uniform(-1.4, 1.4)
        f.vy = rng.uniform(1.6, 3.0)
        f.y += 0.6


def _land(f: Fly, w: float, h: float) -> None:
    if f.y <= 0.0:
        f.y, f.surface, f.vx, f.vy = 0.0, FLOOR, 0.0, 0.0
    elif f.y >= h:
        f.y, f.surface, f.vx, f.vy = h, CEILING, 0.0, 0.0
    elif f.x <= 0.0:
        f.x, f.surface, f.vx, f.vy = 0.0, LEFT, 0.0, 0.0
    elif f.x >= w:
        f.x, f.surface, f.vx, f.vy = w, RIGHT, 0.0, 0.0


def distance(a: Fly, b: Fly) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)
