# quality_gates.py
from __future__ import annotations
from collections import Counter, defaultdict
from typing import List, Tuple, Dict
from protein.helpers import extract_regions
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from anarci import anarci  # HMM‑based antibody numbering

MIN_BIN_SIZE = 5
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


_real_parsed_frs = None  # global cache

def load_real_frameworks(path: str) -> list[dict]:
    global _real_parsed_frs
    if _real_parsed_frs is None:
        df = pd.read_csv(path).fillna("")
        _real_parsed_frs = df.to_dict(orient="records")
    return _real_parsed_frs

def quality_losses(gen_seqs: List[str],
                   real_seqs: List[str],
                   full_real_frs_csv: Optional[str] = None) -> Tuple[float, float]:
    # Use extract_regions to parse generated and real sequences
    gen_parsed = extract_regions(gen_seqs)
    real_parsed = extract_regions(real_seqs)

    # FR loss from full reference set
    if full_real_frs_csv:
        ref_frs = load_real_frameworks(full_real_frs_csv)
        fr_loss = best_framework_loss(gen_parsed, ref_frs)
    else:
        fr_loss = best_framework_loss(gen_parsed, real_parsed)

    cdr_loss = cdr_js_loss(gen_parsed, real_parsed)
    return fr_loss, cdr_loss

# ───────────────────────────────────────────────────────────────
# 2. Framework similarity (identity) loss
# ───────────────────────────────────────────────────────────────
def best_framework_loss(gen_parsed: list[dict], real_parsed: list[dict]) -> float:
    """
    For each generated sequence's full framework (FR1–FR4),
    find the best matching real framework and compute identity loss.

    Args:
        gen_parsed: List of dicts from extract_regions(), one per generated sequence.
        real_parsed: Full list of real framework dicts (preloaded once).

    Returns:
        Mean loss ∈ [0, 1], where 0 = perfect match to best real FR.
    """
    losses = []

    for g in gen_parsed:
        g_fr = g.get("FR1", "") + g.get("FR2", "") + g.get("FR3", "") + g.get("FR4", "")
        if not g_fr:
            losses.append(1.0)
            continue

        best_loss = 1.0
        for r in real_parsed:
            r_fr = r.get("FR1", "") + r.get("FR2", "") + r.get("FR3", "") + r.get("FR4", "")
            if len(r_fr) != len(g_fr):
                continue  # skip mismatched lengths

            mismatches = sum(gc != rc for gc, rc in zip(g_fr, r_fr))
            identity = 1.0 - mismatches / len(g_fr)
            loss = 1.0 - identity
            best_loss = min(best_loss, loss)

        losses.append(best_loss)

    return float(np.mean(losses)) if losses else 1.0


def aa_frequency(rows: list[str]) -> np.ndarray:
    """Converts a list of CDRs (same length) into a [L, 20] frequency matrix."""
    L = len(rows[0])
    freq = np.zeros((L, 20))
    for s in rows:
        for i, aa in enumerate(s):
            if aa in AMINO_ACIDS:
                j = AMINO_ACIDS.index(aa)
                freq[i, j] += 1
    freq = np.where(freq == 0, 1e-9, freq)
    freq /= freq.sum(-1, keepdims=True)
    return freq

def cdr_js_loss(gen_parsed: list[dict], real_parsed: list[dict]) -> float:
    """
    Computes average Jensen-Shannon divergence for each CDR (1–3) across bins grouped by length.

    Args:
        gen_parsed: List of parsed dicts from extract_regions() for generated sequences.
        real_parsed: List of parsed dicts from extract_regions() for real sequences.

    Returns:
        JS loss in [0, 1], lower = more similar distribution.
    """
    losses, weights = [], []

    for cdr in ("CDR1", "CDR2", "CDR3"):
        real_bins = defaultdict(list)
        gen_bins = defaultdict(list)

        # Group sequences by length
        for entry in real_parsed:
            cdr_seq = entry.get(cdr)
            if cdr_seq:
                real_bins[len(cdr_seq)].append(cdr_seq)

        for entry in gen_parsed:
            cdr_seq = entry.get(cdr)
            if cdr_seq:
                gen_bins[len(cdr_seq)].append(cdr_seq)

        # Compute JS divergence per length bin
        for L in real_bins:
            if L in gen_bins and len(real_bins[L]) >= MIN_BIN_SIZE and len(gen_bins[L]) >= MIN_BIN_SIZE:
                real_freq = aa_frequency(real_bins[L])
                gen_freq = aa_frequency(gen_bins[L])
                js = np.mean([
                    jensenshannon(real_freq[i], gen_freq[i])
                    for i in range(L)
                ])
                losses.append(js)
                weights.append(len(real_bins[L]))

    if not losses:
        return 1.0

    return float(np.average(losses, weights=weights))
