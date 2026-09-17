"""The right-hand column: what the brains are actually doing, tick by tick.

This is not colour commentary. Every line it prints is a threshold crossing on
a named population of real neurons, with the number that crossed it. If a line
says DNp01 fired, two cells fired.
"""
from __future__ import annotations

from .fly import ADULT, AIR


class Narrator:
    def __init__(self, viv):
        self.viv = viv
        self.lines: list[tuple] = []
        self.prev: dict[int, dict] = {}
        self._recent: dict[str, int] = {}
        self._gate: dict[tuple, int] = {}
        self.max = 400

    def emit(self, kind: str, text: str) -> None:
        # One fly having a bad minute should not push everything else off the
        # screen, so an identical line inside a short window is dropped.
        tick = self.viv.jar.tick
        last = self._recent.get(text)
        if last is not None and tick - last < 90:
            return
        self._recent[text] = tick
        if len(self._recent) > 300:
            self._recent = {k: v for k, v in self._recent.items() if tick - v < 400}
        self.lines.append((tick, kind, text))
        if len(self.lines) > self.max:
            del self.lines[: self.max // 3]

    def observe(self) -> None:
        jar = self.viv.jar

        # jar-level events raised by the world model
        while jar.events:
            e = jar.events.pop(0)
            self.emit(e.kind, e.text)

        for f in jar.adults():
            if f.slot < 0:
                continue
            p = self.prev.setdefault(f.ident, {})
            r = self.viv.readouts(f)

            self._cross(f, p, r, "escape", 0.40,
                        lambda v: f"{f.name}: DNp01 at {v:.2f} -- Giant Fibre. "
                                  f"She is airborne before she knows it.")
            self._cross(f, p, r, "backward", 0.45,
                        lambda v: f"{f.name}: MDN at {v:.2f}. Walking backwards.")
            self._cross(f, p, r, "freeze", 0.55,
                        lambda v: f"{f.name}: DNp09 at {v:.2f}. Stopped dead.")
            self._cross(f, p, r, "proboscis", 0.30,
                        lambda v: f"{f.name}: proboscis motor at {v:.2f}. "
                                  f"Mouth out.")
            if f.sex == "male":
                self._cross(f, p, r, "song", 0.18,
                            lambda v: f"{f.name}: pIP10 + TN1 at {v:.2f}, wing "
                                      f"motor neurons following. Singing.")
            else:
                self._cross(f, p, r, "courtship", 0.12,
                            lambda v: f"{f.name}: pC1 at {v:.2f}.")

            was_air = p.get("air", False)
            now_air = f.surface == AIR
            if now_air and not was_air:
                self.emit("move", f"{f.name} took off. {f.last_spikes:,} neurons "
                                  f"firing at the time.")
            elif was_air and not now_air:
                self.emit("move", f"{f.name} landed on the {f.surface}.")
            p["air"] = now_air

    def _cross(self, f, p, r, key, thresh, msg) -> None:
        v = r.get(key)
        if v is None:
            return
        from .behaviour import REF
        norm = min(v / REF.get(key, 1.0), 1.0)
        was = p.get(key, 0.0)
        p[key] = norm
        if norm >= thresh > was:
            # Rate-limit per fly per population, not per exact string: the
            # numbers differ every time, so text dedupe never caught a fly
            # that sang once every ten ticks for a minute.
            gate = (f.ident, key)
            last = self._gate.get(gate, -9999)
            if self.viv.jar.tick - last < 120:
                return
            self._gate[gate] = self.viv.jar.tick
            self.emit("neuro", msg(norm))

    def trace(self, f) -> list:
        """A live readout of one fly, for the detail panel."""
        if f.slot < 0:
            return []
        from .behaviour import REF
        co = self.viv.cohorts[f.sex]
        rows = []
        for key in ("descending", "escape", "freeze", "backward", "steer",
                    "proboscis", "courtship", "song", "wing", "legs", "oviposit"):
            if key not in co.wiring.outputs:
                continue
            raw = co.readout(f.slot, key)
            rows.append((key, min(raw / REF.get(key, 1.0), 1.0), len(co.wiring.outputs[key])))
        return rows
