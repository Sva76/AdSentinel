import numpy as np
import pandas as pd

from .cdr import extract_cdrs
from .esm_utils import ESMEmbedder

HYDRO = set(["A", "V", "I", "L", "M", "F", "W", "Y"])

def aa_fraction(seq, aa_set):
    if not seq:
        return 0.0
    return sum(1 for a in seq if a in aa_set) / len(seq)

def net_charge(seq):
    if not seq:
        return 0.0
    pos = seq.count("K") + seq.count("R")
    neg = seq.count("D") + seq.count("E")
    return (pos - neg) / len(seq)

def compute_sequence_features(df: pd.DataFrame) -> pd.DataFrame:

    df = extract_cdrs(df.copy())
    embedder = ESMEmbedder()

    rows = []

    for row in df.itertuples(index=False):

        vh = row.vh_protein_sequence
        vl = row.vl_protein_sequence

        seq_feats = [
            aa_fraction(vh, HYDRO),
            aa_fraction(vl, HYDRO),
            net_charge(vh),
            net_charge(vl),
            len(vh),
            len(vl),
        ]

        cdr_feats = [
            len(row.cdrh1),
            len(row.cdrh2),
            len(row.cdrh3),
            len(row.cdrl1),
            len(row.cdrl2),
            len(row.cdrl3),
        ]

        emb = embedder.embed(vh + vl)

        rows.append(seq_feats + cdr_feats + list(emb))

    return pd.DataFrame(rows)
