"""The jar.

Sealed glass, a floor, whatever you drop in, and however many flies. Everything
in here is a plain physical model -- diffusion, gravity, sugar, rot -- with one
job: to produce sensory input honest enough to be worth feeding into a real
connectome, and to take real descending commands back out.

The CO2 matters more than it looks. A sealed jar accumulates it, every fly
contributes, and 67 neurons apiece are listening for exactly that. Leave the
lid on long enough and the jar drives itself.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from . import fly as flymod
from .fly import (ADULT, AIR, DEAD, EGG, FLOOR, LARVA, PUPA, Fly, distance,
                  founder, hatchling, step_body)

W = 96.0          # jar width, world units
H = 58.0          # jar height
GW, GH = 24, 15   # odour field resolution

GRAVITY = 0.30
CO2_PER_FLY = 0.00002
CO2_VENT = 0.55
FOOD_ODOUR = 1.5
ROT_ODOUR = 2.2
CVA_GAIN = 4.0


@dataclass
class Food:
    x: float
    y: float
    mass: float
    kind: str = "sugar"     # sugar | yeast
    age: int = 0
    radius: float = 7.0     # a drop of sugar water is a puddle, not a point


@dataclass
class Corpse:
    x: float
    y: float
    name: str
    sex: str
    age: int = 0
    mass: float = 1.0


@dataclass
class Event:
    tick: int
    kind: str
    text: str
    who: int = -1


class Jar:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.nrng = np.random.default_rng(seed)
        self.tick = 0

        self.flies: list[Fly] = []
        self.food: list[Food] = []
        self.corpses: list[Corpse] = []
        self.events: list[Event] = []

        self.light = True
        self.co2 = 0.02
        self.temp = 0.5           # 0 cold .. 1 hot, 0.5 is room temperature
        self.humidity = 0.45
        self.shake = 0.0
        self.gaba_block = 1.0     # 1.0 normal; drop it to block inhibition

        self.odour_food = np.zeros((GH, GW), dtype=np.float32)
        self.odour_rot = np.zeros((GH, GW), dtype=np.float32)
        self.odour_cva = np.zeros((GH, GW), dtype=np.float32)

        self._next_id = 1
        self.born = 0
        self.died = 0

    # ------------------------------------------------------------- roster

    def new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def add_founder(self, sex: str) -> Fly:
        f = founder(self.new_id(), sex, self.rng, W, H)
        self.flies.append(f)
        return f

    def living(self) -> list:
        return [f for f in self.flies if f.stage != DEAD]

    def adults(self) -> list:
        return [f for f in self.flies if f.stage == ADULT]

    def log(self, kind: str, text: str, who: int = -1) -> None:
        self.events.append(Event(self.tick, kind, text, who))
        if len(self.events) > 600:
            del self.events[:200]

    # ------------------------------------------------------------- fields

    def _cell(self, x: float, y: float) -> tuple:
        gx = min(GW - 1, max(0, int(x / W * GW)))
        gy = min(GH - 1, max(0, int(y / H * GH)))
        return gy, gx

    def sample(self, field: np.ndarray, x: float, y: float) -> float:
        gy, gx = self._cell(x, y)
        return float(field[gy, gx])

    def _diffuse(self, f: np.ndarray, rate: float, decay: float) -> np.ndarray:
        p = np.pad(f, 1, mode="edge")
        blur = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) * 0.25
        return ((f * (1.0 - rate) + blur * rate) * decay).astype(np.float32)

    def _update_fields(self) -> None:
        self.odour_food[:] = 0.0
        for it in self.food:
            gy, gx = self._cell(it.x, it.y)
            self.odour_food[gy, gx] += FOOD_ODOUR * min(it.mass, 2.0)
            for dx in (-1, 1):
                gy2, gx2 = self._cell(it.x + dx * it.radius, it.y)
                self.odour_food[gy2, gx2] += FOOD_ODOUR * 0.5 * min(it.mass, 2.0)
        self.odour_rot[:] = 0.0
        for c in self.corpses:
            gy, gx = self._cell(c.x, c.y)
            # A fresh corpse smells of nothing. It gets worse.
            ripeness = min(c.age / 900.0, 1.0)
            self.odour_rot[gy, gx] += ROT_ODOUR * ripeness * c.mass

        for _ in range(3):
            self.odour_food = self._diffuse(self.odour_food, 0.55, 0.985)
            self.odour_rot = self._diffuse(self.odour_rot, 0.55, 0.99)
        # cVA is a plume, not a bath: it has to fade when he stops singing,
        # or the whole jar saturates and every nose in it adapts away.
        self.odour_cva = self._diffuse(self.odour_cva, 0.40, 0.93)

    # -------------------------------------------------------------- senses

    def sense(self, f: Fly, cohort, slot: int) -> dict:
        """Everything the jar is currently doing to one fly.

        Returns channel -> intensity. The engine turns these into current in
        the actual receptor neurons for that modality.
        """
        s: dict[str, float] = {}
        g = f.genome

        if self.light:
            # Overhead light. A fly on the ceiling is right under it.
            lit = 0.55 + 0.45 * (f.y / H)
            s["light"] = lit * 0.9 * g["sight"]
        else:
            s["light"] = 0.02

        # Everyone else's wings. Near-field sound falls off fast.
        heard = 0.0
        sung = 0.0
        for o in self.adults():
            if o.ident == f.ident:
                continue
            d = distance(f, o)
            att = 1.0 / (1.0 + 0.16 * d * d / 10.0)
            heard += o.buzz * att
            if o.sex == "male" and f.sex == "female":
                sung += o.song * att
        if heard + sung > 0.004:
            s["hearing"] = min((heard + sung * 1.7), 3.0) * g["hearing"]

        # Smell. Food, rot, and each other.
        food = self.sample(self.odour_food, f.x, f.y)
        rot = self.sample(self.odour_rot, f.x, f.y)
        if food + rot > 0.004:
            s["food_smell"] = min(food + rot * 0.8, 3.0) * g["smell"]

        cva = self.sample(self.odour_cva, f.x, f.y)
        if cva > 0.004:
            # CVA_GAIN converts field units into receptor drive. It is not
            # cosmetic: her pC1 needs about 1.5 at ORN_DA1 before those ten
            # cells say anything, and the concentration where a female actually
            # stands -- a few cells away from a singing male, after diffusion --
            # sits around 0.4. Without this the courtship loop is wired
            # correctly end to end and never fires once.
            s["pheromone"] = min(cva * CVA_GAIN, 3.0) * g["pheromone"]

        if self.co2 > 0.05:
            s["co2"] = min(self.co2 * 2.2, 3.0) * g["smell"]

        # Taste is contact. She has to be standing on it.
        bite = self._food_under(f)
        if bite is not None:
            s["taste"] = 1.7 * g["taste"]

        # Touch: the glass, and each other.
        contact = 0.25 if f.surface != AIR else 0.0
        for o in self.adults():
            if o.ident != f.ident and distance(f, o) < 2.2:
                contact += 0.9
        if self.shake > 0.01:
            contact += self.shake * 2.0
        if contact > 0.01:
            s["touch"] = min(contact, 3.0) * g["touch"]

        if self.shake > 0.01:
            s["jolt"] = min(self.shake * 2.4, 3.0) * g["touch"]
            s["wind"] = min(self.shake * 1.2, 3.0) * g["hearing"]

        if abs(self.temp - 0.5) > 0.08:
            s["heat"] = min(abs(self.temp - 0.5) * 4.0, 3.0)
        if abs(self.humidity - 0.45) > 0.1:
            s["damp"] = min(abs(self.humidity - 0.45) * 3.0, 3.0)

        return s

    def _food_under(self, f: Fly):
        if f.surface != FLOOR:
            return None
        for it in self.food:
            if abs(it.x - f.x) < it.radius * min(1.0, 0.35 + it.mass) and it.mass > 0.01:
                return it
        return None

    # ---------------------------------------------------------------- step

    def step_world(self) -> None:
        self.tick += 1
        self.shake *= 0.80
        n_adult = len(self.adults())

        # Sealed jar. Everyone breathes, nothing leaves.
        self.co2 += CO2_PER_FLY * n_adult
        self.co2 = min(self.co2, 1.6)
        self.humidity = min(1.0, self.humidity + 0.00004 * n_adult)
        self.temp += (0.5 - self.temp) * 0.002

        self._update_fields()

        for it in self.food:
            it.age += 1
            it.mass -= 0.00008
        self.food = [it for it in self.food if it.mass > 0.01]
        for c in self.corpses:
            c.age += 1
            c.mass -= 0.00022
        self.corpses = [c for c in self.corpses if c.mass > 0.02]

    def step_bodies(self) -> None:
        for f in self.flies:
            if f.stage == DEAD:
                continue
            f.age += 1
            self._life_stage(f)
            if f.stage == ADULT:
                step_body(f, W, H, GRAVITY, self.rng)
                self._metabolise(f)
                self._eat(f)
                # A singing male leaves cVA where he is standing. This is the
                # channel that actually matters -- see _mating().
                if f.sex == "male" and f.song > 0.05:
                    gy, gx = self._cell(f.x, f.y)
                    self.odour_cva[gy, gx] += f.song * 0.85
            elif f.stage == LARVA:
                self._larva(f)
        self._mating()
        self._lay()

    def _life_stage(self, f: Fly) -> None:
        if f.stage == EGG and f.age > flymod.EGG_TICKS:
            f.stage = LARVA
            self.log("hatch", f"{f.name} hatched. A larva has no connectome in "
                               f"this jar -- nobody has loaded one.", f.ident)
        elif f.stage == LARVA and f.age > flymod.EGG_TICKS + flymod.LARVA_TICKS:
            f.stage = PUPA
            self.log("pupate", f"{f.name} pupated.", f.ident)
        elif f.stage == PUPA and f.age > (flymod.EGG_TICKS + flymod.LARVA_TICKS
                                          + flymod.PUPA_TICKS):
            f.stage = ADULT
            f.energy = 0.7
            self.log("eclose", f"{f.name} eclosed -- {f.sex}, generation "
                               f"{f.genome.generation}. Needs a brain.", f.ident)

    def _metabolise(self, f: Fly) -> None:
        burn = 0.00016 * f.genome["metabolism"]
        burn *= 1.0 + 2.4 * f.buzz              # flying is expensive
        burn *= 1.0 + 0.8 * max(0.0, self.temp - 0.5) * 2
        f.energy -= burn
        if f.energy <= 0.0:
            self.kill(f, "starved")
        elif f.age > flymod.LIFESPAN:
            self.kill(f, "old age")
        elif self.co2 > 1.35:
            if self.rng.random() < 0.0022:
                self.kill(f, "carbon dioxide")

    def _eat(self, f: Fly) -> None:
        it = self._food_under(f)
        if it is None:
            return
        # The proboscis has to actually be out, and that is a motor command
        # read off the connectome, not a decision the body makes.
        if f.proboscis > 0.22:
            take = min(it.mass, 0.0026)
            it.mass -= take
            f.energy = min(1.35, f.energy + take * 2.6)

    def _larva(self, f: Fly) -> None:
        # Larvae crawl toward food and eat it. No brain simulated: the larval
        # connectome is a different animal (3,016 neurons) and we do not load it.
        best, bd = None, 1e9
        for it in self.food:
            d = abs(it.x - f.x)
            if d < bd:
                best, bd = it, d
        if best is not None:
            f.x += 0.22 * (1 if best.x > f.x else -1)
            if bd < 2.5:
                take = min(best.mass, 0.0022)
                best.mass -= take
                f.energy = min(1.2, f.energy + take * 1.8)
        f.energy -= 0.00009
        if f.energy <= 0.0:
            self.kill(f, "starved as a larva")

    # ------------------------------------------------------------- the point

    def _mating(self) -> None:
        """Courtship, on whatever circuit the graph actually provides.

        This did not end up where it was aimed. The intended loop was the one
        from the literature: he sings, she hears it with Johnston's organ, her
        pC1 cluster decides. His half works exactly as advertised -- pIP10 and
        TN1 drive his wing motor neurons and we can watch the muscles get the
        order, because his dataset has muscles in it.

        Her half does not. Driving her Johnston's organ at any intensity we can
        produce moves her pC1 by precisely nothing. What does move it is cVA,
        through ORN_DA1, and only that. Every other input channel we have --
        sound, wind, touch, light -- leaves those ten cells silent.

        So that is what this uses. He sings, singing deposits cVA where he is
        standing, it diffuses, she smells it, her pC1 responds, and she decides.
        The pathway is real and well documented; it simply is not the one the
        song story would have you draw. We did not add a sound->pC1 shortcut to
        make the demo work.
        """
        males = [f for f in self.adults() if f.sex == "male"]
        females = [f for f in self.adults() if f.sex == "female" and not f.gravid]
        for m in males:
            if m.courted < 0.10 or m.energy < 0.25:
                continue
            for w in females:
                if distance(m, w) > 12.0:
                    continue
                w.courting = m.ident
                if w.receptive > 0.30 and self.rng.random() < 0.08:
                    w.mated = True
                    w.gravid = 1.0
                    w.mate_genome = m.genome
                    m.energy -= 0.05
                    self.log("mate",
                             f"{w.name} accepted {m.name}. His pIP10 peaked at "
                             f"{m.courted:.2f}; her pC1 at {w.receptive:.2f} -- "
                             f"off his cVA, not his song.",
                             w.ident)
                    break

    def _lay(self) -> None:
        """Egg laying.

        Requiring food underneath her, which is what a female prefers, meant no
        egg was ever laid: she has no way to find the food. Odour from the
        vinegar ORNs spreads through about 1,300 neurons and reaches a
        descending neuron at 0.0008, which is not a behaviour, so there is no
        taxis in this jar and a gravid female only ever finds a puddle by
        walking into one. Real flies strongly prefer a moist substrate and will
        still lay on a dry one, so that is what this does -- food is twelve
        times likelier, and the log says which it was.

        The trigger is the body model, not oviDN. oviDN does not fire. Its
        readout is printed next to every egg so you can watch it not fire.
        """
        for f in self.adults():
            if f.sex != "female" or not f.gravid or f.energy < 0.35:
                continue
            if f.surface != FLOOR:
                continue
            it = self._food_under(f)
            p = 0.05 if it is not None else 0.004
            if self.rng.random() >= p:
                continue
            egg = hatchling(self.new_id(), f, getattr(f, "mate_genome", f.genome),
                            self.rng)
            egg.x, egg.y = f.x + self.rng.uniform(-2, 2), 0.0
            self.flies.append(egg)
            f.gravid = 0.0
            f.energy -= 0.14
            self.born += 1
            where = "on the food" if it is not None else "on bare glass"
            self.log("lay",
                     f"{f.name} laid an egg {where}. (Body model, not oviDN -- "
                     f"her oviDN readout is {f.oviposit:.3f}.)",
                     f.ident)

    def kill(self, f: Fly, cause: str) -> None:
        if f.stage == DEAD:
            return
        was = f.stage
        f.die(self.tick, cause)
        self.died += 1
        self.corpses.append(Corpse(f.x, f.y, f.name, f.sex))
        self.log("death", f"{f.name} died: {cause} (as {was}, age {f.age}).",
                 f.ident)

    # ------------------------------------------------------- what you can do

    def feed(self, kind: str = "sugar") -> None:
        x = self.rng.uniform(W * 0.1, W * 0.9)
        self.food.append(Food(x=x, y=0.0, mass=3.0, kind=kind))
        self.log("feed", f"{kind} dropped in at x={x:.0f}.")

    def toggle_light(self) -> None:
        self.light = not self.light
        self.log("light", "Light on." if self.light else "Light off.")

    def vent(self) -> None:
        before = self.co2
        self.co2 *= (1.0 - CO2_VENT)
        self.humidity = max(0.2, self.humidity - 0.12)
        self.log("vent", f"Lid off. CO2 {before:.2f} -> {self.co2:.2f}.")

    def shake_jar(self) -> None:
        self.shake = 1.0
        for f in self.adults():
            if f.surface != AIR and self.rng.random() < 0.55:
                f.surface = AIR
                f.vx = self.rng.uniform(-2.0, 2.0)
                f.vy = self.rng.uniform(0.5, 2.0)
        self.log("shake", "You shook the jar.")

    def warm(self, d: float) -> None:
        self.temp = min(1.0, max(0.0, self.temp + d))
        self.log("temp", f"Temperature now {self.temp:.2f}.")

    def dose(self) -> None:
        """Picrotoxin, more or less: block inhibition and see what is left."""
        if self.gaba_block >= 0.99:
            self.gaba_block = 0.15
            self.log("drug", "Picrotoxin. Inhibition down to 15%. Every GABA "
                             "and glutamate synapse in every fly at once.")
        else:
            self.gaba_block = 1.0
            self.log("drug", "Washed out. Inhibition restored.")
