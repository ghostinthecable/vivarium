"""The thing that owns the jar and the brains and keeps them in step.

One Cohort per sex, because each sex is a different connectome and they cannot
share a matrix. Flies hold a slot index into their cohort's state array; when a
fly dies the slot is released and the next one to eclose gets it, still warm.

Capacity is a hard limit and deliberately so. Every occupied slot is a whole
connectome being integrated, and the jar will tell you when it is full rather
than quietly stop simulating somebody.
"""
from __future__ import annotations

import time

import numpy as np

from . import behaviour, connectome
from .engine import Cohort
from .fly import ADULT, DEAD
from .world import Jar


class Vivarium:
    def __init__(self, capacity: int = 12, seed: int = 0, quiet: bool = False):
        self.capacity = capacity
        self.jar = Jar(seed=seed)
        self.brains = {s: connectome.load(s, quiet=quiet) for s in ("female", "male")}
        self.cohorts = {s: Cohort(self.brains[s], capacity=capacity)
                        for s in ("female", "male")}
        self.nrng = np.random.default_rng(seed)
        self.paused = False
        self.rate = 0.0
        self._t_last = time.time()
        self._acc = 0.0
        self.noise = 0.0   # recruitment jitter already varies the input; full-field
        #                    gaussian noise cost more than it bought
        self.overflow = 0

    # ------------------------------------------------------------- roster

    def attach(self, f) -> bool:
        """Give a newly-adult fly a connectome to run on."""
        if f.slot >= 0:
            return True
        co = self.cohorts[f.sex]
        s = co.claim()
        if s < 0:
            self.overflow += 1
            return False
        f.slot = s
        co.configure(s, f.genome)
        return True

    def detach(self, f) -> None:
        if f.slot >= 0:
            self.cohorts[f.sex].release(f.slot)
            f.slot = -1

    def populate(self, females: int, males: int) -> None:
        for _ in range(females):
            self.attach(self.jar.add_founder("female"))
        for _ in range(males):
            self.attach(self.jar.add_founder("male"))
        self.jar.log("start",
                     f"{females} female and {males} male founders sealed in. "
                     f"Her brain is 144,837 neurons and stops at the neck; his "
                     f"is 166,700 and goes all the way to the muscles.")

    def add(self, sex: str):
        f = self.jar.add_founder(sex)
        if not self.attach(f):
            self.jar.log("full", f"No slot for {f.name} -- {self.capacity} "
                                 f"brains is the limit. She is in the jar but "
                                 f"not being simulated.")
        else:
            self.jar.log("add", f"{f.name} introduced ({sex}).")
        return f

    # ---------------------------------------------------------------- tick

    def tick(self) -> None:
        if self.paused:
            return
        jar = self.jar
        jar.step_world()

        for co in self.cohorts.values():
            co.clear_drive()

        # Everything the jar is doing to everybody, into real receptor neurons.
        for f in jar.adults():
            if f.slot < 0:
                continue
            co = self.cohorts[f.sex]
            for ch, amount in jar.sense(f, co, f.slot).items():
                co.stimulate(f.slot, ch, amount, self.nrng)

        for co in self.cohorts.values():
            co.step(gaba_block=jar.gaba_block, noise=self.noise, rng=self.nrng)

        for f in jar.adults():
            if f.slot >= 0:
                behaviour.apply(f, self.cohorts[f.sex])

        jar.step_bodies()

        # Newly eclosed adults need a brain; the dead give theirs back.
        for f in jar.flies:
            if f.stage == ADULT and f.slot < 0:
                self.attach(f)
            elif f.stage == DEAD and f.slot >= 0:
                self.detach(f)

        now = time.time()
        dt = now - self._t_last
        self._t_last = now
        if dt > 0:
            inst = 1.0 / dt
            self.rate = inst if self.rate == 0 else self.rate * 0.9 + inst * 0.1

    # --------------------------------------------------------------- views

    def brain_line(self, f) -> str:
        if f.slot < 0:
            return "not simulated"
        return behaviour.summary(f, self.cohorts[f.sex])

    def top_dn(self, f, k: int = 6) -> list:
        if f.slot < 0:
            return []
        return self.cohorts[f.sex].top_descending(f.slot, k)

    def readouts(self, f) -> dict:
        if f.slot < 0:
            return {}
        co = self.cohorts[f.sex]
        return {k: co.readout(f.slot, k) for k in co.wiring.outputs}

    def slots_used(self) -> tuple:
        return (int(self.cohorts["female"].live.sum()),
                int(self.cohorts["male"].live.sum()))
