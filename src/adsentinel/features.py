AdSentinel 2.7 – Feature pipeline with Click support.

Three feature blocks built separately for the Click mechanism:
  Block 1 (Global):  sequence physicochemical + isotype
  Block 2 (Zoom):    CDR-specific descriptors (hydropathy, charge, patches)
  Block 3 (ESM):     ESM-2 mean-pooled embeddings (1280-dim)
"""

import numpy as np
import pandas as pd
from pathlib import Path


# --- Constants ---

HYDRO = set("AVILMFWY")

KD_SCALE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}

# AHo alignment CDR positions (0-indexed, derived from gap analysis on GDPa1)
CDR_RANGES_HEAVY = {
    'CDR-H1': (24, 40),
    'CDR-H2': (58, 77),
    'CDR-H3': (107, 138),
}
CDR_RANGES_LIGHT = {
    'CDR-L1': (24, 40),
    'CDR-L2': (56, 66),
    'CDR-L3': (107, 135),
}

ESM_VECTORS_PATH = Path(__file__).parent.parent.parent / "data" / "AbSentinel_vectors_1280.csv"


# --- Helper functions ---

def aa_fraction(seq, aa_set):
    if not isinstance(seq, str) or len(seq) == 0:
        return 0.0
    return sum(1 for a in seq if a in aa_set) / len(seq)


def net_charge(seq):
    if not isinstance(seq, str) or len(seq) == 0:
        return 0.0
    pos = seq.count("K") + seq.count("R")
    neg = seq.count("D") + seq.count("E")
    return (pos - neg) / len(seq)


def extract_region(aligned_seq, start, end):
    """Extract residues (no gaps) from AHo-aligned positions."""
    region = aligned_seq[start:end]
    return ''.join(c for c in region if c != '-')


def region_hydropathy_stats(seq):
    """Mean, max, std of Kyte-Doolittle hydropathy for a region."""
    if not seq or len(seq) == 0:
        return 0, 0, 0
    values = [KD_SCALE.get(aa, 0) for aa in seq]
    return np.mean(values), np.max(values), np.std(values)


def region_charge_stats(seq):
    """Net charge, absolute charge density, and length for a region."""
    if not seq or len(seq) == 0:
        return 0, 0, 0
    pos = sum(1 for aa in seq if aa in 'KRH')
    neg = sum(1 for aa in seq if aa in 'DE')
    net = (pos - neg) / len(seq)
    abs_charge = (pos + neg) / len(seq)
    return net, abs_charge, len(seq)


def sliding_window_max_hydro(seq, window=5):
    """Detect stickiest patch: max mean hydropathy in sliding window."""
    if not seq or len(seq) < window:
        return 0
    values = [KD_SCALE.get(aa, 0) for aa in seq]
    return max(
        np.mean(values[i:i + window])
        for i in range(len(values) - window + 1)
    )


def load_esm_vectors(path=None):
    """Load pre-computed ESM-2 mean-pooled embeddings."""
    p = Path(path) if path else ESM_VECTORS_PATH
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df.set_index("label")
    return df


# --- Feature block builders ---

def build_global_features(df):
    """
    Block 1: Global sequence features + isotype.
    Returns numpy array (n_samples, ~9).
    """
    rows = []
    for _, row in df.iterrows():
        vh = row.get("vh_protein_sequence", "")
        vl = row.get("vl_protein_sequence", "")
        rows.append([
            len(vh), len(vl),
            aa_fraction(vh, HYDRO), aa_fraction(vl, HYDRO),
            net_charge(vh), net_charge(vl),
        ])

    X = np.array(rows)

    if "hc_subtype" in df.columns:
        iso = pd.get_dummies(
            df["hc_subtype"].fillna("UNK").astype(str),
            prefix="isotype",
        ).values.astype(float)
        X = np.hstack([X, iso])

    return X


def build_zoom_features(df):
    """
    Block 2: CDR-specific zoom features.
    Per CDR: length, hydropathy (mean/max/std), charge (net/abs), sticky patch.
    Returns numpy array (n_samples, 42).
    """
    rows = []
    for _, row in df.iterrows():
        z = []
        h_aligned = row.get("heavy_aligned_aho", "")
        l_aligned = row.get("light_aligned_aho", "")

        for ranges, aligned in [(CDR_RANGES_HEAVY, h_aligned), (CDR_RANGES_LIGHT, l_aligned)]:
            if not isinstance(aligned, str) or len(aligned) == 0:
                z.extend([0] * 7 * len(ranges))
                continue
            for name, (s, e) in ranges.items():
                region = extract_region(aligned, s, e)
                hm, hx, hs = region_hydropathy_stats(region)
                nc, ac, length = region_charge_stats(region)
                patch = sliding_window_max_hydro(region)
                z.extend([length, hm, hx, hs, nc, ac, patch])

        rows.append(z)

    return np.array(rows, dtype=float)


def build_esm_features(df, esm_path=None):
    """
    Block 3: ESM-2 embeddings (1280-dim).
    Returns numpy array (n_samples, 1280) or None if embeddings not found.
    """
    esm_df = load_esm_vectors(esm_path)
    if esm_df is None or "antibody_name" not in df.columns:
        return None

    rows = []
    matched = 0
    for name in df["antibody_name"]:
        if name in esm_df.index:
            rows.append(esm_df.loc[name].values.astype(float))
            matched += 1
        else:
            rows.append(np.zeros(esm_df.shape[1]))

    print(f"[ESM] Matched {matched}/{len(df)} antibodies")
    return np.array(rows)


# --- Legacy interface (backward compatible) ---

def compute_sequence_features(df, esm_path=None):
    """
    Build full feature matrix (all blocks concatenated).
    Used by train_cv.py and predict.py when Click is not active.
    """
    X_global = build_global_features(df)
    X_zoom = build_zoom_features(df)
    X_esm = build_esm_features(df, esm_path)

    parts = [X_global, X_zoom]
    if X_esm is not None:
        parts.append(X_esm)

    X = np.hstack(parts)

    n_g = X_global.shape[1]
    n_z = X_zoom.shape[1]
    n_e = X_esm.shape[1] if X_esm is not None else 0
    print(f"[FEATURES] Global: {n_g} | CDR zoom: {n_z} | ESM: {n_e} | Total: {X.shape[1]}")

    return pd.DataFrame(X)
