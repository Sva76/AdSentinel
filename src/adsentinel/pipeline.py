import pandas as pd
import numpy as np
from .features import aa_fraction, net_charge, HYDRO, POS, NEG, AROM
from .cdr import extract_cdrs
from .esm_utils import esm_embed_sequence
from .model import AdSentinelModel

def extract_all_features(df):
    feats = []

    for i, row in df.iterrows():
        vh = row["vh_protein_sequence"]
        vl = row["vl_protein_sequence"]

        seq_feats = [
            aa_fraction(vh, HYDRO),
            aa_fraction(vl, HYDRO),
            net_charge(vh),
            net_charge(vl),
            len(vh),
            len(vl)
        ]

        cdrs = extract_cdrs(row.to_frame().T)
        cdr_feats = []
        for c in cdrs:
            cdr_feats.append(len(row[c]))

        emb = esm_embed_sequence(vh + vl)

        feats.append(seq_feats + cdr_feats + list(emb))

    return np.array(feats)

def train_adsentinel(train_df, target):
    X = extract_all_features(train_df)
    y = train_df[target].values
    model = AdSentinelModel()
    model.fit(X, y)
    return model

def predict_adsentinel(model, df):
    X = extract_all_features(df)
    return model.predict(X)
