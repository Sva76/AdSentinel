import numpy as np
import pandas as pd

HYDRO = set(["A", "V", "I", "L", "M", "F", "W", "Y"])


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


def compute_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build AdSentinel feature matrix.

    Features:
    - simple physicochemical sequence features
    - VH/VL length
    - hydrophobic fraction
    - net charge
    - antibody isotype (hc_subtype) one-hot encoding
    """

    rows = []

    for row in df.itertuples(index=False):

        vh = getattr(row, "vh_protein_sequence", "")
        vl = getattr(row, "vl_protein_sequence", "")

        seq_feats = [
            len(vh),
            len(vl),
            aa_fraction(vh, HYDRO),
            aa_fraction(vl, HYDRO),
            net_charge(vh),
            net_charge(vl),
        ]

        rows.append(seq_feats)

    feat_df = pd.DataFrame(
        rows,
        columns=[
            "vh_length",
            "vl_length",
            "vh_hydrophobic_fraction",
            "vl_hydrophobic_fraction",
            "vh_net_charge",
            "vl_net_charge",
        ],
    )

    # ----------------------------
    # Isotype feature (Ginkgo tip)
    # ----------------------------

    if "hc_subtype" in df.columns:

        isotype_df = pd.get_dummies(
            df["hc_subtype"].fillna("UNK").astype(str),
            prefix="isotype",
        )

        feat_df = pd.concat([feat_df, isotype_df], axis=1)

    return feat_df.astype(float)

Add hc_subtype (isotype) feature to AdSentinel pipeline
