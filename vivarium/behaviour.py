"""Turning descending-neuron firing rates into things a body does.

The reference scales below are not invented. Each is the peak decayed readout
that population reached when every input channel in channels.py was driven at
intensity 1.6 for 200 ticks, measured once against both connectomes. Dividing
by them puts every command on a comparable 0..1 footing, so "her giant fibre is
at 0.7" means the same as "his wing motor neurons are at 0.7".

Two of these came back flat, and both are kept rather than papered over:

  oviposit  never fires. Not once, from any stimulus we can deliver. oviDN is
            six cells buried deep enough that nothing in our input repertoire
            reaches them. Egg laying in this jar is therefore driven by the
            body model -- gravid, standing on food -- and the log says so every
            single time. The oviDN readout is still displayed, still honest,
            and still zero.

  steer     fires in her and not in him. DNa01/DNa02 are present in both
            connectomes and only hers light up. We do not know why. The male
            steers off his generic descending drive instead, and the roster
            marks it.
"""
from __future__ import annotations

# A population that fired flat out every tick would settle at 1/(1-0.82) = 5.6
# on the decayed readout; the refractory period caps a real one nearer 1.9. The
# numbers below sit between the measured peak for each population and that cap,
# which keeps two-cell populations like DNp01 from pinning at 1.00 the moment
# they fire at all -- they did, at first, and the fly spent her whole life
# bouncing off the ceiling.
REF = {
    "descending": 0.28,
    "escape": 1.70,
    "freeze": 1.40,
    "backward": 0.60,
    "steer": 0.50,
    "proboscis": 0.44,
    "courtship": 0.17,
    "song": 0.31,
    "wing": 0.32,
    "legs": 0.19,
    "oviposit": 0.02,
}


def _norm(v: float, key: str) -> float:
    r = REF.get(key, 1.0)
    x = v / r if r else 0.0
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def apply(fly, cohort) -> None:
    """Read one fly's brain and set what its body is doing."""
    s = fly.slot
    if s < 0:
        return
    g = fly.genome

    fly.last_spikes = int(cohort.spiked[s])

    dn = _norm(cohort.readout(s, "descending"), "descending")
    fly.drive = dn
    fly.escape = _norm(cohort.readout(s, "escape"), "escape") * g["nerve"]
    fly.freeze = _norm(cohort.readout(s, "freeze"), "freeze")
    fly.backward = _norm(cohort.readout(s, "backward"), "backward")
    fly.proboscis = _norm(cohort.readout(s, "proboscis"), "proboscis")
    # Receptivity is a state, not a firing rate. Her pC1 answers a cVA plume in
    # a burst three or four ticks long and then adaptation closes it; holding
    # the decaying peak is what makes it a mood rather than a twitch. Without
    # this she was above threshold for 7 ticks out of 1600, and every one of
    # them landed while the nearest male was just too far away.
    _pc1 = _norm(cohort.readout(s, "courtship"), "courtship") * g["lust"]
    fly.receptive = max(fly.receptive * 0.99, _pc1)

    if fly.sex == "male":
        # He has a nerve cord, so we can watch the wing muscles themselves.
        wing = _norm(cohort.readout(s, "wing"), "wing")
        fly.song = _norm(cohort.readout(s, "song"), "song") * g["lust"]
        fly.buzz = max(wing, dn * 0.6)
        # Courtship is a state, not an instant. The cVA he leaves while
        # singing has to diffuse to her and then cross her brain before her
        # pC1 says anything, and by then he has usually stopped. Holding a
        # decaying trace of "has been singing lately" is what makes the loop
        # close; requiring him to be mid-note at the moment she decides was
        # double-counting the same causality and mating never once happened.
        fly.courted = max(fly.courted * 0.985, fly.song)
        fly.oviposit = 0.0
        steer = dn                                  # DNa01/02 stay silent in him
    else:
        # Her dataset stops at the neck. There are no wing motor neurons to
        # read, so the buzz is inferred from descending drive alone. This is
        # the one place the two sexes are not measured the same way.
        fly.buzz = dn * 0.75
        fly.song = 0.0
        fly.oviposit = _norm(cohort.readout(s, "oviposit"), "oviposit")
        steer = _norm(cohort.readout(s, "steer"), "steer")

    # Turning: steering drive past a threshold flips which way she is facing.
    if steer > 0.55:
        fly.heading = -fly.heading


def summary(fly, cohort) -> str:
    """One line of what the brain is doing, for the log."""
    s = fly.slot
    if s < 0:
        return f"{fly.name}: no brain attached ({fly.stage})"
    bits = [f"{cohort.spiked[s]:5d} firing"]
    for k in ("escape", "freeze", "backward", "song", "wing", "oviposit"):
        v = cohort.readout(s, k)
        if v > 0.001:
            bits.append(f"{k}={_norm(v, k):.2f}")
    return "  ".join(bits)
