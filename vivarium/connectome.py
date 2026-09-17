"""Two real brains, normalised into one shape the jar can use.

  female   FlyWire FAFB v783    144,837 neurons. Brain only. Stops at the neck.
  male     Janelia MaleCNS v1.0 166,691 neurons. Brain AND ventral nerve cord.

They were traced by different labs, a decade apart, with different software and
completely different naming conventions. Almost all of the work in this file is
making their annotation columns agree without inventing anything that isn't in
one of them.

The asymmetry survives that process on purpose. He has wing motor neurons and
leg motor neurons and she does not, because his dataset goes down there and
hers does not. When she issues a descending command it stops at the bottom of
her file. That is not a bug in this program, it is the edge of the map.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import scipy.sparse as sp

from . import data

CACHE_VERSION = 3

# Standard connectome-sim approximation: sign a neuron's outgoing synapses by
# its predicted transmitter. Acetylcholine excites. GABA inhibits. Glutamate
# inhibits, in flies, via GluCl-alpha -- do not carry that to a mouse. The
# monoamines do not fit a fast +/- weight at all, so they get zero rather than
# a guess.
SIGNS = {
    "acetylcholine": +1.0,
    "glutamate": -1.0,
    "gaba": -1.0,
    "dopamine": 0.0,
    "serotonin": 0.0,
    "octopamine": 0.0,
    "histamine": -1.0,     # photoreceptor transmitter, inhibitory onto LMCs
    "unknown": 0.0,
}

# --------------------------------------------------------------- normalising

# FAFB super_class -> our role
_ROLE_F = {
    "descending": "descending",
    "motor": "motor",
    "sensory": "sensory",
    "sensory_ascending": "sensory",
    "sensory_descending": "sensory",
    "ascending": "ascending",
}
# MaleCNS superclass -> our role
_ROLE_M = {
    "descending_neuron": "descending",
    "vnc_motor": "motor",
    "cb_motor": "motor",
    "vnc_efferent": "motor",
    "cb_efferent": "motor",
    "cb_sensory": "sensory",
    "ol_sensory": "sensory",
    "vnc_sensory": "sensory",
    "sensory_ascending": "sensory",
    "sensory_descending": "sensory",
    "ascending_neuron": "ascending",
}

# FAFB cell_function -> our modality
_MOD_F = {
    "visual_achromatic": "visual",
    "visual_chromatic": "visual",
    "visual_ocellar": "visual",
    "visual_polarized_light": "visual",
    "polarized_light": "visual",
    "olfactory": "olfactory",
    "gustatory": "gustatory",
    "tactile": "tactile",
    "auditory": "auditory",
    "proprioception": "proprioceptive",
    "thermosensory": "thermo",
    "hygrosensory": "hygro",
    "proboscis_motor": "proboscis_motor",
    "unknown_mechanosensory": "tactile",
}
# MaleCNS class -> our modality
_MOD_M = {
    "visual": "visual",
    "olfactory": "olfactory",
    "gustatory": "gustatory",
    "mechanosensory_tactile": "tactile",
    "mechanosensory": "tactile",
    "mechanosensory_tbc": "tactile",
    "mechanosensory_proprioceptive": "proprioceptive",
    "thermosensory": "thermo",
    "hygrosensory": "hygro",
    "chemosensory": "olfactory",
}


@dataclass
class Brain:
    sex: str
    n: int
    ids: np.ndarray
    Wexc: sp.csr_matrix         # [post, pre], positive weights only
    Winh: sp.csr_matrix         # [post, pre], negative weights only
    celltype: np.ndarray        # native cell type string
    flywiretype: np.ndarray     # FlyWire-convention type, for cross-sex lookup
    role: np.ndarray            # descending | motor | sensory | ascending | ""
    modality: np.ndarray        # visual | olfactory | ... | ""
    subclass: np.ndarray        # male only, else ""
    side: np.ndarray
    has_vnc: bool

    def select(self, *, celltype_in=None, celltype_prefix=None,
               flywire_prefix=None, role=None, modality=None,
               subclass_in=None, exclude_prefix=None) -> np.ndarray:
        m = np.ones(self.n, dtype=bool)
        if celltype_in is not None:
            want = set(celltype_in)
            m &= np.isin(self.celltype, list(want)) | np.isin(self.flywiretype, list(want))
        if celltype_prefix is not None:
            pres = tuple(celltype_prefix)
            m &= (_startswith(self.celltype, pres) | _startswith(self.flywiretype, pres))
        if flywire_prefix is not None:
            m &= _startswith(self.flywiretype, tuple(flywire_prefix))
        if role is not None:
            m &= self.role == role
        if modality is not None:
            m &= self.modality == modality
        if subclass_in is not None:
            m &= np.isin(self.subclass, list(subclass_in))
        if exclude_prefix is not None:
            m &= ~(_startswith(self.celltype, tuple(exclude_prefix)))
        return np.flatnonzero(m)


def _startswith(arr: np.ndarray, prefixes: tuple) -> np.ndarray:
    out = np.zeros(len(arr), dtype=bool)
    for p in prefixes:
        out |= np.char.startswith(arr, p)
    return out


def _cache(sex: str) -> Path:
    return data.data_dir() / f"{sex}_brain_v{CACHE_VERSION}.npz"


# ------------------------------------------------------------------- loading

def load(sex: str, rebuild: bool = False, quiet: bool = False) -> Brain:
    return _load_female(rebuild, quiet) if sex == "female" else _load_male(rebuild, quiet)


def _finish(sex, ids, w, pre_i, post_i, n, celltype, flywiretype, role,
            modality, subclass, side, has_vnc, cache, quiet):
    """Normalise weights by each neuron's total input, split by sign, cache."""
    # Share of the postsynaptic cell's whole input budget. Keeps a neuron with
    # 20,000 inputs from being 100x louder than one with 200.
    tot = np.bincount(post_i, weights=np.abs(w), minlength=n).astype(np.float32)
    tot[tot == 0] = 1.0
    wn = (w / tot[post_i]).astype(np.float32)

    pos, neg = wn > 0, wn < 0
    Wexc = sp.csr_matrix((wn[pos], (post_i[pos], pre_i[pos])), shape=(n, n), dtype=np.float32)
    Winh = sp.csr_matrix((wn[neg], (post_i[neg], pre_i[neg])), shape=(n, n), dtype=np.float32)
    Wexc.sum_duplicates(); Winh.sum_duplicates()

    np.savez_compressed(
        cache, n=np.int64(n), ids=ids,
        edata=Wexc.data, eind=Wexc.indices, eptr=Wexc.indptr,
        idata=Winh.data, iind=Winh.indices, iptr=Winh.indptr,
        celltype=celltype, flywiretype=flywiretype, role=role,
        modality=modality, subclass=subclass, side=side,
    )
    if not quiet:
        print(f"    {sex}: {n:,} neurons, {Wexc.nnz:,} excitatory + "
              f"{Winh.nnz:,} inhibitory edges", file=sys.stderr)
    return Brain(sex, n, ids, Wexc, Winh, celltype, flywiretype, role,
                 modality, subclass, side, has_vnc)


def _from_cache(sex: str, cache: Path, has_vnc: bool) -> Brain:
    z = np.load(cache, allow_pickle=False)
    n = int(z["n"])
    Wexc = sp.csr_matrix((z["edata"], z["eind"], z["eptr"]), shape=(n, n))
    Winh = sp.csr_matrix((z["idata"], z["iind"], z["iptr"]), shape=(n, n))
    return Brain(sex, n, z["ids"], Wexc, Winh,
                 z["celltype"].astype(str), z["flywiretype"].astype(str),
                 z["role"].astype(str), z["modality"].astype(str),
                 z["subclass"].astype(str), z["side"].astype(str), has_vnc)


def _load_female(rebuild: bool, quiet: bool) -> Brain:
    cache = _cache("female")
    if cache.exists() and not rebuild:
        return _from_cache("female", cache, has_vnc=False)

    paths = data.ensure_sex("female")
    if not quiet:
        print("  building female connectome (one-time) ...", file=sys.stderr)
    mt = feather.read_table(paths["female_meta"])
    ids = pc.cast(mt.column("fafb_783_id"), "int64").to_numpy(zero_copy_only=False).astype(np.int64)
    order = np.argsort(ids)
    ids = ids[order]
    n = len(ids)

    def col(name):
        a = np.asarray(mt.column(name).to_pylist(), dtype=object)[order]
        return np.array(["" if x is None else str(x) for x in a], dtype=object).astype(str)

    celltype = col("cell_type")
    sup = col("super_class")
    fn = col("cell_function")
    side = col("side")
    role = np.array([_ROLE_F.get(s, "") for s in sup], dtype=object).astype(str)
    modality = np.array([_MOD_F.get(f, "") for f in fn], dtype=object).astype(str)
    subclass = np.array([""] * n, dtype="<U1")
    nt = col("neurotransmitter_predicted")

    et = feather.read_table(paths["female_edges"])
    pre = pc.cast(et.column("pre"), "int64").to_numpy(zero_copy_only=False).astype(np.int64)
    post = pc.cast(et.column("post"), "int64").to_numpy(zero_copy_only=False).astype(np.int64)
    norm = et.column("norm").to_numpy(zero_copy_only=False).astype(np.float32)

    pre_i = np.searchsorted(ids, pre)
    post_i = np.searchsorted(ids, post)
    ok = ((pre_i < n) & (post_i < n)
          & (ids[np.clip(pre_i, 0, n - 1)] == pre)
          & (ids[np.clip(post_i, 0, n - 1)] == post))
    pre_i, post_i, norm = pre_i[ok], post_i[ok], norm[ok]

    sign = np.array([SIGNS.get(x, 0.0) for x in nt], dtype=np.float32)
    w = norm * sign[pre_i]
    keep = w != 0
    return _finish("female", ids, w[keep], pre_i[keep], post_i[keep], n,
                   celltype, celltype.copy(), role, modality, subclass, side,
                   False, cache, quiet)


def _load_male(rebuild: bool, quiet: bool) -> Brain:
    cache = _cache("male")
    if cache.exists() and not rebuild:
        return _from_cache("male", cache, has_vnc=True)

    paths = data.ensure_sex("male")
    if not quiet:
        print("  building male connectome (one-time, ~2 min, 152M raw edges) ...",
              file=sys.stderr)

    at = feather.read_table(paths["male_ann"])
    ad = at.to_pydict()
    body = np.asarray(ad["bodyId"], dtype=np.int64)
    sup_raw = _strs(ad["superclass"])

    # A body with no superclass is an unclassified fragment, not a neuron.
    # Dropping them lands on 166,691, which is the published neuron count.
    real = sup_raw != ""
    ids = body[real]
    order = np.argsort(ids)
    ids = ids[order]
    n = len(ids)
    pick = np.flatnonzero(real)[order]

    celltype = _strs(ad["type"])[pick]
    flywiretype = _strs(ad["flywireType"])[pick]
    subclass = _strs(ad["subclass"])[pick]
    side = _strs(ad["somaSide"])[pick]
    sup = sup_raw[real][order]
    cls = _strs(ad["class"])[pick]
    role = np.array([_ROLE_M.get(s, "") for s in sup], dtype=object).astype(str)
    modality = np.array([_MOD_M.get(c, "") for c in cls], dtype=object).astype(str)

    # JO neurons are filed as mechanosensory here and auditory in FAFB. They
    # are the same organ. Make them agree.
    jo = _startswith(celltype, ("JO-",)) | _startswith(flywiretype, ("JO-",))
    modality[jo] = "auditory"

    nt_tbl = feather.read_table(paths["male_nt"], columns=["body", "consensus_nt"])
    nt_body = nt_tbl.column("body").to_numpy(zero_copy_only=False).astype(np.int64)
    nt_val = _strs(nt_tbl.column("consensus_nt").to_pylist())
    sign_by_body = np.zeros(n, dtype=np.float32)
    j = np.searchsorted(ids, nt_body)
    good = (j < n) & (ids[np.clip(j, 0, n - 1)] == nt_body)
    sign_by_body[j[good]] = [SIGNS.get(v, 0.0) for v in nt_val[good]]

    # 152M raw segment-to-segment rows, streamed a batch at a time so this
    # never needs 4GB resident.
    pres, posts, ws = [], [], []
    with pa.memory_map(str(paths["male_weights"]), "r") as src:
        reader = pa.ipc.open_file(src)
        for b in range(reader.num_record_batches):
            rb = reader.get_batch(b)
            p = rb.column("body_pre").to_numpy(zero_copy_only=False).astype(np.int64)
            q = rb.column("body_post").to_numpy(zero_copy_only=False).astype(np.int64)
            v = rb.column("weight").to_numpy(zero_copy_only=False).astype(np.float32)
            pi = np.searchsorted(ids, p)
            qi = np.searchsorted(ids, q)
            ok = ((pi < n) & (qi < n)
                  & (ids[np.clip(pi, 0, n - 1)] == p)
                  & (ids[np.clip(qi, 0, n - 1)] == q))
            if not ok.any():
                continue
            pi, qi, v = pi[ok], qi[ok], v[ok]
            s = sign_by_body[pi]
            nz = s != 0
            if nz.any():
                pres.append(pi[nz]); posts.append(qi[nz]); ws.append(v[nz] * s[nz])

    pre_i = np.concatenate(pres); post_i = np.concatenate(posts); w = np.concatenate(ws)
    return _finish("male", ids, w, pre_i, post_i, n, celltype, flywiretype,
                   role, modality, subclass, side, True, cache, quiet)


def _strs(seq) -> np.ndarray:
    return np.array(["" if x is None else str(x) for x in seq], dtype=object).astype(str)
