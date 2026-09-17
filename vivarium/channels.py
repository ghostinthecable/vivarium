"""The wiring between the jar and the brains in it.

Every channel here names a real, annotated population in the dataset. Nothing
is approximated into existence: if a population does not exist in one sex's
connectome, the channel is absent for that sex and the jar says so out loud
rather than quietly substituting something similar.

IN  -- what the jar can do to a fly. Each one injects current into the actual
       receptor neurons for that modality.
OUT -- what a fly can do about it. Descending neurons are the brain's outbox.
       In the male we can follow them further, into the nerve cord, and watch
       the wing muscles themselves get the order.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_BOTH = object()


@dataclass(frozen=True)
class Spec:
    """A selector, per sex. `None` means 'this sex does not have these'."""
    female: dict | None
    male: dict | None
    label: str
    gene: str = ""              # genome gain applied to this channel
    note: str = ""

    def for_sex(self, sex: str) -> dict | None:
        return self.female if sex == "female" else self.male


def _both(d: dict, label: str, gene: str = "", note: str = "") -> Spec:
    return Spec(female=d, male=d, label=label, gene=gene, note=note)


# --------------------------------------------------------------------- input

INPUTS: dict[str, Spec] = {
    "light": _both(
        dict(modality="visual"), "photoreceptors", "sight",
        "11,148 retinal cells in her, 6,091 in him -- the datasets photographed "
        "different amounts of eye."),
    "hearing": _both(
        dict(celltype_prefix=("JO-A", "JO-B")), "Johnston's organ, sound", "hearing",
        "JO-A and JO-B are the sound-sensitive subgroups. Wingbeats and song "
        "land here."),
    "wind": _both(
        dict(celltype_prefix=("JO-C", "JO-D", "JO-E", "JO-F")),
        "Johnston's organ, wind and gravity", "hearing",
        "The same antennal organ, the subgroups that do steady deflection "
        "rather than vibration."),
    "food_smell": _both(
        dict(celltype_in=("ORN_DM1", "ORN_DM2")), "vinegar-line ORNs", "smell",
        "DM1 and DM2 are the acetic acid and ethyl-ester glomeruli. This is "
        "what rotting fruit smells like to a fly."),
    "pheromone": _both(
        dict(celltype_in=("ORN_DA1",)), "ORN_DA1, the cVA line", "pheromone",
        "126 neurons in her, 204 in him. cVA is transferred by males during "
        "mating; both sexes smell it and disagree about what it means."),
    "co2": _both(
        dict(celltype_in=("ORN_V",)), "ORN_V, carbon dioxide", "smell",
        "67 neurons. Stress signal. In a sealed jar it only goes up."),
    "taste": _both(
        dict(modality="gustatory"), "gustatory neurons", "taste",
        "459 in her, 1,428 in him -- his dataset includes the leg taste "
        "bristles, so he can taste what he is standing on and she cannot."),
    "touch": _both(
        dict(modality="tactile"), "bristle neurons", "touch"),
    "heat": _both(
        dict(modality="thermo"), "thermosensory neurons", "",
        "33 cells in her, 25 in him. Barely a thermometer."),
    "damp": _both(
        dict(modality="hygro"), "hygrosensory neurons", "",
        "Humidity. 74 cells in her, 66 in him."),
    "jolt": _both(
        dict(modality="proprioceptive"), "proprioceptors", "touch",
        "What being shaken feels like from inside."),
}


# -------------------------------------------------------------------- output

OUTPUTS: dict[str, Spec] = {
    "escape": _both(
        dict(celltype_in=("DNp01",)), "DNp01, the Giant Fibre", "nerve",
        "Two cells. The single best-understood neuron in the animal. It fires, "
        "she is airborne, nobody consulted her."),
    "backward": _both(
        dict(celltype_in=("MDN",)), "MDN, moonwalker", "vigour",
        "Four cells. Drive them and the fly walks backwards. This is not a "
        "metaphor, it is the experiment."),
    "freeze": _both(
        dict(celltype_in=("DNp09",)), "DNp09, stop", "vigour"),
    "steer": _both(
        dict(celltype_in=("DNa01", "DNa02")), "DNa01/DNa02, turning", "vigour"),
    "descending": _both(
        dict(role="descending"), "all descending neurons", "vigour",
        "The whole outbox: 1,301 cells in her, 1,314 in him. Everything the "
        "brain is currently telling the body."),
    "courtship": _both(
        dict(celltype_prefix=("pC1",)), "pC1 cluster", "lust",
        "10 cells in her, 156 in him -- the male cluster is subdivided far "
        "more finely, and the ones called P1 in the literature are in there."),

    # ---- male only, because his dataset has a body attached
    "song": Spec(
        female=None,
        male=dict(celltype_prefix=("pIP10", "TN1")),
        label="pIP10 + TN1, courtship song",
        gene="lust",
        note="pIP10 is the descending neuron that commands song; TN1 are the "
             "pattern generators in the nerve cord that produce it. She has "
             "neither, and does not sing."),
    "wing": Spec(
        female=None,
        male=dict(role="motor", subclass_in=("wm",)),
        label="wing motor neurons",
        gene="vigour",
        note="67 cells: DLMn and DVMn for power, b1/b2/b3/hg for steering and "
             "song. Absent from her dataset entirely -- FAFB stops at the neck."),
    "legs": Spec(
        female=None,
        male=dict(role="motor", subclass_in=("fl", "ml", "hl")),
        label="leg motor neurons",
        gene="vigour",
        note="381 cells across front, middle and hind legs."),

    # ---- female only, because the male doesn't lay
    "oviposit": Spec(
        female=dict(celltype_prefix=("oviDN",)),
        male=None,
        label="oviDN, egg laying",
        gene="",
        note="Six cells. The descending command to lay an egg. There is no "
             "male equivalent and the male dataset has none."),
    "proboscis": Spec(
        female=dict(modality="proboscis_motor"),
        male=dict(role="motor", subclass_in=("pm",)),
        label="proboscis motor neurons",
        gene="taste",
        note="Sticking the mouth out at something. Annotated differently in "
             "the two datasets; both are real motor neurons to the mouthparts."),
}


@dataclass
class Wiring:
    """Resolved neuron indices for one brain."""
    sex: str
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    absent: list = field(default_factory=list)

    def has(self, key: str) -> bool:
        return key in self.inputs or key in self.outputs


def resolve(brain) -> Wiring:
    w = Wiring(sex=brain.sex)
    for name, table, dest in (("in", INPUTS, w.inputs), ("out", OUTPUTS, w.outputs)):
        for key, spec in table.items():
            sel = spec.for_sex(brain.sex)
            if sel is None:
                w.absent.append((key, spec.label, "not in this sex's connectome"))
                continue
            idx = brain.select(**sel)
            if len(idx) == 0:
                w.absent.append((key, spec.label, "selector matched nothing"))
                continue
            dest[key] = idx
    return w


def census(brain, wiring: Wiring) -> list:
    rows = []
    for key, idx in wiring.inputs.items():
        rows.append(("in", key, INPUTS[key].label, len(idx)))
    for key, idx in wiring.outputs.items():
        rows.append(("out", key, OUTPUTS[key].label, len(idx)))
    return rows
