# vivarium

A sealed jar. Some flies in it. The flies are real fly brains.

Every adult in the jar is a whole connectome being integrated on your machine,
a dozen at a time. The females are FlyWire FAFB v783 — 144,837 neurons traced
from one female fruit fly's head by electron microscopy. The males are Janelia
MaleCNS v1.0 — 166,700 neurons from a male, brain *and* ventral nerve cord.

Nothing talks to a language model. Nobody narrates. You drop sugar in, take the
lid off, turn the light out, and watch what 300,000 real neurons do about it.

```
================================================     COMPUTATION
|                                              |      412  Ostella: pIP10 + TN1 at 0.33, wing
|~o~                                           |           motor neurons following. Singing.
|                                              |      418  Pyxilla: DNp01 at 0.56 -- Giant
|              \o/                             |           Fibre. She is airborne before she
|                                              |           knows it.
|                    ~o-                       |      431  Palara accepted Droope. His pIP10
| ~o~   ::::::::~o~        ~o~     :::::  ~o.  |           peaked at 0.22; her pC1 at 0.50 --
+----------------------------------------------+           off his cVA, not his song.

 Hesandra -- female, generation 0
 stage adult  age 1702  energy 0.57
 6,234 of 144,837 neurons firing (4.30%)

 descending  ####............  0.27   1301c
 escape      #...............  0.06      2c
 oviposit    ................  0.00      6c
```

## The two brains are not the same brain

This is the whole reason the project exists and it is not a detail.

|                  | female (FAFB v783) | male (MaleCNS v1.0) |
|------------------|--------------------|---------------------|
| neurons          | 144,837            | 166,700             |
| signed edges     | 14.1M              | 24.6M               |
| coverage         | brain only         | brain + nerve cord  |
| photoreceptors   | 19,250             | 6,091               |
| gustatory        | 459                | 1,428               |
| motor neurons    | 106                | 913                 |
| wing muscles     | **none in dataset**| 67                  |
| leg muscles      | **none in dataset**| 381                 |
| `pIP10` (song)   | **absent**         | 2                   |
| `oviDN` (laying) | 6                  | **absent**           |

FAFB stops at the neck. She has 1,301 descending neurons — the brain's entire
outbox — and in her dataset there is nothing on the other end of them. When she
decides to move a leg, the decision is real and the leg is off the edge of the
map. He has the nerve cord, so when his brain orders a wing muscle to fire you
can watch the muscle get the order.

Her wingbeat is therefore *inferred* from descending drive, and his is *read off
his motor neurons*. That is the one place the two sexes are not measured the
same way, and it is because of what was photographed, not because of anything
this program decided.

## Setup

```bash
git clone https://github.com/ghostinthecable/vivarium.git
cd vivarium
./setup.sh
./bin/jar
```

That is the whole thing. `setup.sh` makes a venv, installs three packages
(numpy, scipy, pyarrow), downloads ~1.5GB of fly from two public Google Cloud
buckets, and builds the connectivity matrices. It takes a few minutes, mostly
downloading.

**No login, no API key, no credentials of any kind.** Both datasets are
published open; neither wants an account. Nothing here phones home.

You need:

- **python 3.10 or newer** — no other system packages
- **about 2.1GB of disk** — 1.5GB of connectome plus ~600MB of cached matrices
- **a terminal at least 60x18** — it will tell you if yours is smaller
- a network connection, for the first run only

`setup.sh` is safe to re-run. Interrupted downloads resume where they stopped
and verify their length before being accepted, which matters because the male
edge list alone is 1.1GB.

```bash
./setup.sh --check      # report what is already in place, change nothing
./setup.sh --no-data    # venv only, fetch the connectomes later
./setup.sh --rebuild    # discard the cached matrices and rebuild
```

Then:

```bash
./bin/jar --census      # what resolved in each connectome, and what didn't
./bin/jar --selftest    # re-measure every claim this README makes
./bin/jar --bench 200   # headless, report ticks/sec
```

If you already have a [flyclaude](https://github.com/ghostinthecable/flyclaude)
checkout, the female half is symlinked rather than downloaded twice. It looks in
`/opt/flybrain/data`; point `VIVARIUM_FLYBRAIN` somewhere else if yours lives
elsewhere. `VIVARIUM_DATA` moves where everything is stored.

### If something goes wrong

**`could not create a venv`** — on Debian and Ubuntu the venv module ships
separately: `sudo apt install python3-venv`.

**The download died partway.** Run `./setup.sh` again. It resumes. If a file
somehow arrives the wrong length it is refused rather than used, and the partial
is kept for the next attempt.

**`terminal too small`** — resize to at least 60x18. It redraws as soon as you
do; nothing crashes.

**It is slow.** Each adult fly is a whole connectome being integrated. Fewer
flies is the dial: `./bin/jar -f 2 -m 2`. `--bench` tells you what your machine
does.

## Using it

```bash
./bin/jar                 # 3 females, 3 males
./bin/jar -f 6 -m 6       # more of them
./bin/jar -n 20           # raise the simulated-brain limit
./bin/jar --dark          # start with the light off
```

While it runs:

```
f  drop sugar in          l  light on/off        v  take the lid off (vents CO2)
s  shake the jar          d  picrotoxin          h/c  warmer / colder
a  add a fly              m  add a male          e  add a female
K  kill the selected fly  TAB  select next       x  dump readout to the log
SPACE pause               ?  help                q  quit
```

Light on draws the jar black-on-white, light off white-on-black, because that
is the actual difference the flies are responding to.

```
~o~   walking on the glass        \o/  /o\   in the air, wings beating
~o/   a male singing -- one wing out, which is what courtship song is
~o.   proboscis out               -x-  dead
.     egg     ~  larva     (o)  pupa     :::::  food
```

Flying flies flap. A singing male holds one wing out and vibrates it, because
that is what *Drosophila* courtship song physically is: unilateral wing
extension. You can watch him do it from across the room.

## Things that came out of the graph, not out of me

**She does not respond to his song.** The intended courtship loop was the one
from the literature: he sings, she hears it with Johnston's organ, her pC1
cluster decides. His half works exactly as advertised — pIP10 and TN1 drive his
wing motor neurons and you can watch the muscles get the order. Her half does
not. Driving her Johnston's organ at any intensity this program can produce
moves her pC1 by *precisely nothing*. What does move it is cVA, through
ORN_DA1, and only that — every other channel leaves those ten cells silent. So
that is what mating runs on: he sings, singing deposits cVA where he stands, it
diffuses, she smells it, her pC1 responds. The pathway is real and well
documented. It is simply not the one the song story would have you draw, and
there is no sound→pC1 shortcut in here to make the demo work.

**oviDN never fires.** Not once, from any stimulus this jar can deliver. Six
cells, too deep. Egg-laying is therefore driven by the body model, and the log
says so explicitly every single time it happens, printing the oviDN readout
next to the egg so you can watch it stay at 0.000.

**Nothing in the jar can find food.** Requiring a gravid female to stand on
food before laying — which is what a female prefers — produced zero eggs in
five thousand ticks, every time. She has no way to locate a puddle: the odour
does spread, through about 1,300 neurons, but it reaches a descending neuron at
0.0008, which is not a behaviour. There is no taxis in here. Flies find food by
walking into it. Real females strongly prefer a moist substrate and will still
lay on a dry one, so laying on bare glass is twelve times less likely than
laying on food rather than impossible, and the log says which it was.

**DNa01/DNa02 steer her and not him.** Both pairs are present in both
connectomes, identically named. Hers answer the photoreceptors — visually
guided turning, which is what those cells are for — and answer nothing else.
His answer nothing this jar can deliver: not light, touch, hearing, wind or
being shaken. I don't know why. He steers off generic descending drive instead
and the roster says so.

**Smelling food doesn't move her.** Activity from the vinegar ORNs spreads
through 1,300 neurons and reaches a descending neuron 93 ticks later at
0.0008 — effectively never. Flies in this jar find food by walking into it and
tasting it, because taste reaches the motor neurons and smell doesn't.

All of the above is checked by `./bin/jar --selftest`, which re-measures every
claim in this section against the datasets and fails loudly if one stops being
true. It caught me overstating the steering result while I was writing it.

## How much of this is real

**The wiring is real.** Both graphs were traced from real animals that were
sliced up and photographed. `MDN` really does make a fly walk backwards.
`DNp01` is the Giant Fibre and it is genuinely why you keep missing.

**The dynamics are a cartoon.** Leaky integrate-and-fire: charge accumulates,
neuron pops, charge resets by subtraction, brief refractory period. Real neurons
have dendrites, synaptic delays, neuropeptides and opinions. These have a float
and a threshold. Every project in this space makes this trade; this one is
being loud about it.

Two things had to be added to stop the model being obviously wrong, and both are
real properties of real sense organs rather than fudges:

- **Sensory adaptation.** Without it, 19,250 photoreceptors under a constant
  light pin the Giant Fibre at maximum forever and the fly spends her entire
  life bouncing off the ceiling. It did exactly that at first.
- **A giant-fibre refractory period.** The real escape circuit habituates hard
  to repeated looming, which is why you eventually get the fly with the glass.
- **Courtship and receptivity as states rather than firing rates.** His song
  and her pC1 answer are both bursts a few ticks long. The cVA he leaves has to
  diffuse to her and then cross her brain, by which time he has stopped
  singing, so requiring both at the same instant double-counted the same
  causality and mating never happened once in thousands of ticks. Both sides
  now hold a decaying trace of their own peak. The readouts displayed are still
  the instantaneous ones.

**The genome is gains, not wiring.** Flies inherit a short list of knobs —
excitability, per-channel sensory gain, metabolism, nerve — and the measured
graph underneath never changes. Rewiring 25 million edges per fly would cost
100MB a head and, worse, would be fiction: nobody has measured what a rewired
fly's connectome looks like. Most real behavioural variation between flies is
receptor expression and gain anyway.

**Larvae have no brain here.** The larval connectome is a different animal
(3,016 neurons, separately published) and this program does not load it. Larvae
crawl toward food on three lines of arithmetic and the log says so when they
hatch.

**They aren't conscious.** They're graphs. Very, very good graphs.

## Performance

One sparse matrix per sex, shared by every fly of that sex — the wiring is the
same measured wiring, that's the point — so what differs between flies is only
their state and their gains. State is one dense `(neurons × flies)` array and a
tick is a single sparse-times-dense product per cohort. Twelve flies cost barely
more than one.

`csr_matvecs` is single-threaded, so the matrices are split into row blocks and
run on a four-thread pool; the blocks write to disjoint output rows so there is
nothing to synchronise. Measured ~2.5x. More than four threads loses to memory
bandwidth.

On a 20-core box: **~12.8 ticks/sec with 6 flies, ~10.3 with 16.** The simulation
runs on its own thread and the screen redraws at a fixed 15fps regardless,
because stepping twelve whole brains takes about 80ms and a UI that waited for
it would feel broken.

## Data

Both datasets are public and neither wants a login.

- **FlyWire FAFB v783**, CC BY-SA 4.0. Dorkenwald et al. (2024), *Neuronal
  wiring diagram of an adult brain*, Nature; Schlegel et al. (2024),
  *Whole-brain annotation and multi-connectome cell typing of Drosophila*,
  Nature.
- **Janelia FlyEM MaleCNS v1.0**, CC BY 4.0. Released October 2025 — the first
  complete male *Drosophila* central nervous system, brain and nerve cord, with
  fruitless/doublesex expression annotated.
  <https://male-cns.janelia.org/>

Neither is committed here. `.gitignore` covers the lot.

If you do anything real with this, cite them. They mapped two brains and I put
them in a jar.

## FAQ

**Are the flies okay?**
One has been dead since about 2018 and the other since about 2023. They are now
1.5GB of feather files. Doing better than most of us.

**Why is the male the one with legs?**
Because that's the dataset that has legs in it. Ask Janelia.

**Can I make them play DOOM?**
The readout is already descending-neuron commands, which is exactly what you'd
map to buttons. Go be the eighteenth person to do it.

**Something interesting happened and I can't reproduce it.**
`--seed`. Though the flies drift apart within a few hundred ticks regardless,
because sensory recruitment is stochastic and so is everything downstream of it.

## Licence

The code is MIT — see [LICENSE](LICENSE). Copyright 2026 Dan.

The connectomes are not mine and carry their own terms, above: FAFB is
CC BY-SA 4.0, MaleCNS is CC BY 4.0. Nothing in `data/` is redistributed here.
`setup.sh` fetches both straight from the publishers' own public buckets, and
`.gitignore` covers the lot, so the MIT licence applies to the program and
never to the brains it loads.

One thing to know if you fork this and start shipping artifacts: the built
matrices, `data/female_brain_v3.npz`, are an adaptation of CC BY-SA material.
Keeping them local is fine and is what happens by default. Putting one in a
release, a container image or a dataset upload means that artifact goes out
under CC BY-SA 4.0 with attribution, whatever this file says about the code.

## Prior art

- [flyclaude](https://github.com/ghostinthecable/flyclaude) — the same female
  brain, having a conversation instead of a life
- [doomfly](https://github.com/nftechie/doomfly), fly connectome plays DOOM
- [flyvis](https://github.com/TuragaLab/flyvis), the serious version
- [flygym](https://github.com/NeLy-EPFL/flygym), the very serious version
