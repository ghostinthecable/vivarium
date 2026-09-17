"""The loop you actually watch.

The simulation runs on its own thread at whatever rate the connectomes allow,
and the screen redraws at a fixed 15fps regardless. Those two things must not
be tied together: stepping twelve whole brains takes about eighty milliseconds
and a UI that waited for it would feel broken.
"""
from __future__ import annotations

import curses
import threading
import time

from . import render
from .fly import ADULT, DEAD
from .narrate import Narrator
from .sim import Vivarium

FPS = 15.0

# Below this the panels cannot be laid out at all and curses raises rather than
# clipping. Checked every frame, because terminals get resized.
MIN_COLS, MIN_ROWS = 60, 18


class App:
    def __init__(self, viv: Vivarium):
        self.viv = viv
        self.narrator = Narrator(viv)
        self.lock = threading.Lock()
        self.running = True
        self.selected = -1
        self.show_help = False
        self._pick_first()

    def _pick_first(self) -> None:
        a = self.viv.jar.adults()
        if a:
            self.selected = a[0].ident

    def selected_fly(self):
        for f in self.viv.jar.flies:
            if f.ident == self.selected:
                return f
        return None

    def follow_living(self) -> None:
        """If whoever was selected has died, move to somebody who hasn't."""
        f = self.selected_fly()
        if f is None or f.stage == DEAD:
            self.cycle(+1)

    def cycle(self, step: int) -> None:
        a = [f for f in self.viv.jar.flies if f.stage != DEAD]
        if not a:
            self.selected = -1
            return
        ids = [f.ident for f in a]
        if self.selected in ids:
            i = (ids.index(self.selected) + step) % len(ids)
        else:
            i = 0
        self.selected = ids[i]

    # ----------------------------------------------------------- sim thread

    def sim_loop(self) -> None:
        while self.running:
            if self.viv.paused:
                time.sleep(0.05)
                continue
            with self.lock:
                self.viv.tick()
                self.narrator.observe()
                # Flies die of their own accord too, not just when you press K.
                self.follow_living()

    # ---------------------------------------------------------------- keys

    def handle(self, ch: int) -> None:
        jar = self.viv.jar
        if self.show_help:
            self.show_help = False
            return
        if ch in (ord("q"), 27):
            self.running = False
        elif ch == ord("?"):
            self.show_help = True
        elif ch == ord(" "):
            self.viv.paused = not self.viv.paused
        elif ch == ord("f"):
            with self.lock:
                jar.feed()
        elif ch == ord("l"):
            with self.lock:
                jar.toggle_light()
        elif ch == ord("v"):
            with self.lock:
                jar.vent()
        elif ch == ord("s"):
            with self.lock:
                jar.shake_jar()
        elif ch == ord("d"):
            with self.lock:
                jar.dose()
        elif ch == ord("h"):
            with self.lock:
                jar.warm(+0.12)
        elif ch == ord("c"):
            with self.lock:
                jar.warm(-0.12)
        elif ch == ord("w"):
            with self.lock:
                jar.humidity = min(1.0, jar.humidity + 0.15)
                jar.log("water", f"Humidity now {jar.humidity:.2f}.")
        elif ch in (ord("a"), ord("m"), ord("e")):
            sex = ("male" if ch == ord("m") else
                   "female" if ch == ord("e") else
                   ("male" if self.viv.nrng.random() < 0.5 else "female"))
            with self.lock:
                f = self.viv.add(sex)
            self.selected = f.ident
        elif ch == ord("K"):
            f = self.selected_fly()
            if f is not None and f.stage != DEAD:
                with self.lock:
                    jar.kill(f, "you")
                # Do not leave the cursor parked on a corpse.
                self.cycle(+1)
        elif ch in (9, ord("n")):
            self.cycle(+1)
        elif ch == ord("p"):
            self.cycle(-1)
        elif ch == ord("x"):
            f = self.selected_fly()
            if f is not None:
                with self.lock:
                    self._dump(f)

    def _dump(self, f) -> None:
        n = self.narrator
        n.emit("dump", f"--- {f.name} ({f.sex}, gen {f.genome.generation}) ---")
        if f.slot < 0:
            n.emit("dump", "  not simulated")
            return
        n.emit("dump", f"  {f.last_spikes:,} neurons firing")
        for key, val, ncells in n.trace(f):
            n.emit("dump", f"  {key:<11s} {val:5.3f}  ({ncells} cells)")
        for k, v, d in f.genome.notable(4):
            n.emit("dump", f"  gene {k:<11s} {v:5.2f}  (wild type {d:.2f})")

    # --------------------------------------------------------------- render

    def draw(self, stdscr) -> None:
        if self.show_help:
            render.draw_help(stdscr)
            return
        h, w = stdscr.getmaxyx()
        if w < MIN_COLS or h < MIN_ROWS:
            stdscr.erase()
            msg = f"terminal too small: need {MIN_COLS}x{MIN_ROWS}, have {w}x{h}"
            try:
                stdscr.addstr(0, 0, msg[: max(0, w - 1)])
                if h > 2:
                    stdscr.addstr(2, 0, "resize, or q to quit"[: max(0, w - 1)])
            except curses.error:
                pass
            stdscr.refresh()
            return
        jar_w = max(28, min(int(w * 0.46), w - 34))
        right_w = w - jar_w
        status_h = 2
        body_h = h - status_h
        detail_h = min(16, max(8, body_h // 2))
        jar_h = body_h - detail_h

        stdscr.erase()
        stdscr.noutrefresh()

        with self.lock:
            f = self.selected_fly()
            jw = curses.newwin(jar_h, jar_w, 0, 0)
            render.draw_jar(jw, self.viv, self.selected)
            dw = curses.newwin(detail_h, jar_w, jar_h, 0)
            render.draw_detail(dw, self.viv, self.narrator, f)
            lw = curses.newwin(body_h, right_w, 0, jar_w)
            render.draw_log(lw, self.narrator, self.viv)
            sw = curses.newwin(status_h, w, body_h, 0)
            render.draw_status(sw, self.viv, self.narrator, self.viv.paused)

        curses.doupdate()


def run(viv: Vivarium) -> None:
    app = App(viv)

    def main(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.keypad(True)
        render.init_colours()
        t = threading.Thread(target=app.sim_loop, daemon=True)
        t.start()
        frame = 1.0 / FPS
        while app.running:
            t0 = time.time()
            ch = stdscr.getch()
            while ch != -1:
                app.handle(ch)
                if not app.running:
                    break
                ch = stdscr.getch()
            if not app.running:
                break
            app.draw(stdscr)
            rest = frame - (time.time() - t0)
            if rest > 0:
                time.sleep(rest)
        app.running = False

    try:
        curses.wrapper(main)
    finally:
        app.running = False
