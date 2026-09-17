"""Drawing the jar.

Light on is drawn as black-on-white and light off as white-on-black, because
that is the actual difference the flies are responding to and it should be the
first thing you notice from across the room.
"""
from __future__ import annotations

import curses

from .fly import ADULT, AIR, CEILING, DEAD, EGG, FLOOR, LARVA, LEFT, PUPA, RIGHT
from .world import GH, GW, H, W

C_DEFAULT, C_FEMALE, C_MALE, C_FOOD, C_DEAD, C_DIM, C_HOT, C_OK = range(1, 9)


def init_colours() -> None:
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(C_DEFAULT, -1, -1)
    curses.init_pair(C_FEMALE, curses.COLOR_MAGENTA, -1)
    curses.init_pair(C_MALE, curses.COLOR_CYAN, -1)
    curses.init_pair(C_FOOD, curses.COLOR_YELLOW, -1)
    curses.init_pair(C_DEAD, curses.COLOR_RED, -1)
    curses.init_pair(C_DIM, curses.COLOR_BLUE, -1)
    curses.init_pair(C_HOT, curses.COLOR_RED, -1)
    curses.init_pair(C_OK, curses.COLOR_GREEN, -1)


# A fly is three characters: two wings and a body. Flying flies flap, and a
# singing male holds one wing out, which is what courtship song actually looks
# like -- unilateral wing extension, vibrated. The wings are drawn dim and the
# body in the sex colour, so at a glance the jar reads as insects rather than
# punctuation.
def sprite(f, phase: int):
    """Return (left_wing, body, right_wing) for one fly."""
    if f.stage == DEAD:
        return ("-", "x", "-")
    if f.stage == EGG:
        return ("", ".", "")
    if f.stage == LARVA:
        return ("", "~", "")
    if f.stage == PUPA:
        return ("(", "o", ")")

    if f.surface == AIR:
        # wingbeat
        return ("\\", "o", "/") if phase % 2 else ("/", "o", "\\")
    if f.sex == "male" and f.song > 0.15:
        # one wing out, vibrating
        return ("~", "o", "/") if phase % 2 else ("~", "o", "-")
    if f.proboscis > 0.30:
        return ("~", "o", ".")
    return ("~", "o", "~")


def glyph(f) -> str:
    return sprite(f, 0)[1]


def _safe(win, y, x, text, attr=0) -> None:
    h, w = win.getmaxyx()
    if y < 0 or y >= h or x >= w:
        return
    if x < 0:
        text = text[-x:]
        x = 0
    room = w - x
    if room <= 0:
        return
    try:
        win.addstr(y, x, text[:room], attr)
    except curses.error:
        pass


def draw_jar(win, viv, selected: int) -> None:
    """The jar, side on, with everything currently in it."""
    win.erase()
    h, w = win.getmaxyx()
    jar = viv.jar

    lit = jar.light
    base = curses.A_REVERSE if lit else curses.A_NORMAL

    inner_w = max(8, w - 4)
    inner_h = max(4, h - 4)
    top = 1

    # Glass. The lid sits on top and is the thing you take off to vent.
    lid = "=" * (inner_w + 2)
    _safe(win, 0, 1, lid, base | curses.A_BOLD)
    for row in range(inner_h):
        _safe(win, top + row, 1, "|" + " " * inner_w + "|", base)
    _safe(win, top + inner_h, 1, "+" + "-" * inner_w + "+", base | curses.A_BOLD)

    def place(wx, wy):
        cx = 2 + int(wx / W * (inner_w - 1))
        cy = top + inner_h - 1 - int(wy / H * (inner_h - 1))
        return cy, cx

    # Food and corpses sit on the floor.
    for it in jar.food:
        cy, cx = place(it.x, 0.0)
        ch = ":" if it.mass > 0.5 else "."
        span = it.radius * min(1.0, 0.35 + it.mass)
        half = max(0, int(span / W * (inner_w - 1)))
        for dx in range(-half, half + 1):
            x = cx + dx
            if 1 < x < 1 + inner_w:
                _safe(win, cy, x, ch,
                      base | curses.color_pair(C_FOOD) | curses.A_BOLD)
    for c in jar.corpses:
        cy, cx = place(c.x, 0.0)
        _safe(win, cy, cx - 1, "-x-"[: 3], base | curses.color_pair(C_DEAD))

    # Everybody else.
    phase = jar.tick // 2
    for f in jar.flies:
        if f.stage == DEAD:
            continue
        cy, cx = place(f.x, f.y)
        col = C_FEMALE if f.sex == "female" else C_MALE
        body_attr = base | curses.color_pair(col) | curses.A_BOLD
        wing_attr = base | curses.color_pair(C_DIM)
        if f.stage != ADULT:
            body_attr = base | curses.color_pair(C_DIM)
            wing_attr = body_attr
        if f.ident == selected:
            body_attr |= curses.A_UNDERLINE
            wing_attr |= curses.A_BOLD
        lw, body, rw = sprite(f, phase)
        # Wings are only drawn where there is glass to spare, so a fly walking
        # the left wall does not scribble through it.
        if lw and cx - 1 > 1:
            _safe(win, cy, cx - 1, lw, wing_attr)
        if rw and cx + 1 < 1 + inner_w:
            _safe(win, cy, cx + 1, rw, wing_attr)
        _safe(win, cy, cx, body, body_attr)

    label = " light on " if lit else " light off "
    _safe(win, 0, max(1, w - len(label) - 2), label, base | curses.A_BOLD)
    win.noutrefresh()


def draw_log(win, narrator, viv) -> None:
    win.erase()
    h, w = win.getmaxyx()
    _safe(win, 0, 0, "COMPUTATION".ljust(w - 1), curses.A_REVERSE | curses.A_BOLD)

    kinds = {
        "neuro": C_OK, "death": C_DEAD, "mate": C_FEMALE, "lay": C_FEMALE,
        "feed": C_FOOD, "drug": C_HOT, "shake": C_HOT, "light": C_DIM,
    }
    rows = narrator.lines[-(h - 1):]
    for i, (tick, kind, text) in enumerate(rows):
        y = 1 + i
        if y >= h:
            break
        col = curses.color_pair(kinds.get(kind, C_DEFAULT))
        _safe(win, y, 0, f"{tick:6d} ", curses.color_pair(C_DIM))
        _safe(win, y, 7, text[: max(0, w - 8)], col)
    win.noutrefresh()


def bar(v: float, width: int, ch: str = "#") -> str:
    v = 0.0 if v < 0 else 1.0 if v > 1 else v
    n = int(round(v * width))
    return ch * n + "." * (width - n)


def draw_detail(win, viv, narrator, f) -> None:
    win.erase()
    h, w = win.getmaxyx()
    if f is None:
        _safe(win, 0, 0, "nobody selected".ljust(w - 1),
              curses.A_REVERSE | curses.A_BOLD)
        win.noutrefresh()
        return

    col = C_DEAD if f.stage == DEAD else (C_FEMALE if f.sex == "female" else C_MALE)
    dead = "  (dead)" if f.stage == DEAD else ""
    head = f" {f.name} -- {f.sex}, generation {f.genome.generation}{dead} "
    _safe(win, 0, 0, head.ljust(w - 1),
          curses.A_REVERSE | curses.A_BOLD | curses.color_pair(col))

    brain = viv.brains[f.sex]
    pct = 100.0 * f.last_spikes / brain.n if brain.n else 0.0
    y = 1
    _safe(win, y, 0, f"stage {f.stage:<6s} age {f.age:<6d} energy {f.energy:4.2f}")
    y += 1
    _safe(win, y, 0, f"{f.last_spikes:,} of {brain.n:,} neurons firing ({pct:.2f}%)")
    y += 1
    if f.slot < 0:
        # Be specific about *why* there is no brain here. Saying "at capacity"
        # for a fly that is simply dead, or still an egg, is a lie that looks
        # like a bug.
        if f.stage == DEAD:
            why = f"dead since tick {f.dead_at} -- {f.cause}"
        elif f.stage in (EGG, LARVA, PUPA):
            why = f"a {f.stage} has no connectome loaded in this jar"
        else:
            why = "no free brain slot -- raise it with -n"
        _safe(win, y, 0, why[: w - 1], curses.color_pair(C_HOT))
        win.noutrefresh()
        return

    y += 1
    _safe(win, y, 0, "descending / motor readout", curses.A_BOLD)
    y += 1
    for key, val, ncells in narrator.trace(f):
        if y >= h - 1:
            break
        c = C_HOT if val > 0.6 else C_OK if val > 0.15 else C_DIM
        _safe(win, y, 0, f"{key:<10s}")
        _safe(win, y, 11, bar(val, min(16, max(4, w - 30))),
              curses.color_pair(c))
        _safe(win, y, 11 + min(16, max(4, w - 30)) + 1,
              f"{val:4.2f} {ncells:>5d}c", curses.color_pair(C_DIM))
        y += 1

    if y < h - 1:
        y += 1
        hot = viv.top_dn(f, 4)
        if hot:
            _safe(win, y, 0, "firing now: " + ", ".join(
                f"{n}x{c}" for n, c in hot)[: w - 13], curses.color_pair(C_OK))
            y += 1
    if y < h - 1:
        notable = ", ".join(f"{k} {v:.2f}" for k, v, _d in f.genome.notable(3))
        _safe(win, y, 0, f"genome: {notable}"[: w - 1], curses.color_pair(C_DIM))
    win.noutrefresh()


def draw_status(win, viv, narrator, paused: bool) -> None:
    win.erase()
    h, w = win.getmaxyx()
    jar = viv.jar
    nf = sum(1 for f in jar.adults() if f.sex == "female")
    nm = sum(1 for f in jar.adults() if f.sex == "male")
    brood = sum(1 for f in jar.living() if f.stage != ADULT)

    co2c = C_HOT if jar.co2 > 0.9 else C_OK
    parts = [
        f" {nf}F {nm}M",
        f"brood {brood}",
        f"born {jar.born}/died {jar.died}",
        f"CO2 {jar.co2:4.2f}",
        f"{jar.temp:4.2f}T",
        f"{viv.rate:4.1f} t/s",
    ]
    if jar.gaba_block < 0.99:
        parts.append("PICROTOXIN")
    if paused:
        parts.append("PAUSED")
    line = "  ".join(parts)
    _safe(win, 0, 0, line.ljust(w - 1), curses.A_REVERSE | curses.A_BOLD)
    keys = (" f feed  l light  v vent  s shake  d drug  a/m/e add  K kill "
            " TAB next  SPACE pause  ? help  q quit")
    _safe(win, 1, 0, keys[: w - 1], curses.color_pair(C_DIM))
    win.noutrefresh()


HELP = [
    "VIVARIUM -- a sealed jar with real fly brains in it",
    "",
    "  f   drop sugar in            l   light on / off",
    "  v   take the lid off (vents CO2)",
    "  s   shake the jar            d   picrotoxin: block inhibition",
    "  h/c warmer / colder          w   add water (humidity)",
    "",
    "  a   add a fly (random sex)   m   add a male",
    "  e   add a female             K   kill the selected fly",
    "",
    "  TAB / n   select next fly    p   select previous",
    "  SPACE     pause              x   dump selected fly's readout to the log",
    "  q         quit",
    "",
    "  ~o~  on the glass      \\o/  in the air, wings beating",
    "  ~o/  male singing (one wing out, which is what song looks like)",
    "  ~o.  proboscis out      -x-  dead      :  food",
    "  .    egg      ~  larva      (o)  pupa",
    "",
    "  Her brain is FlyWire FAFB v783: 144,837 neurons, brain only, stops at",
    "  the neck. His is Janelia MaleCNS v1.0: 166,700 neurons, brain and nerve",
    "  cord, so his wing and leg muscles are in the simulation and hers are",
    "  not. That asymmetry is in the data, not in this program.",
    "",
    "  press any key",
]


def draw_help(stdscr) -> None:
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    for i, line in enumerate(HELP):
        if i + 1 >= h:
            break
        attr = curses.A_BOLD if i == 0 else 0
        _safe(stdscr, i + 1, 2, line[: w - 3], attr)
    stdscr.refresh()
