import pandas as pd
import numpy as np

from .features import aa_fraction, net_charge, HYDRO
from .cdr import extract_cdrs
from .esm_utils import esm_embed_sequence
from .model import AdSentinelRegressor


def extract_all_features(df: pd.DataFrame) -> np.ndarray:
    """
    Estrae tutte le feature per VH + VL:
    - proprietà globali di sequenza (frazione aa, carica, lunghezza)
    - lunghezze CDR corrette
    - embedding ESM-2 (mean pooling)
    """

    feats = []

    # --- calcolo CDR una sola volta su tutto il DF ---
    # extract_cdrs deve restituire un DataFrame con colonne:
    # cdrh1, cdrh2, cdrh3, cdrl1, cdrl2, cdrl3
    df_cdr = extract_cdrs(df.copy())

    for row in df_cdr.itertuples(index=False):

        vh = row.vh_protein_sequence
        vl = row.vl_protein_sequence

        # --- feature di sequenza ---
        seq_feats = [
            aa_fraction(vh, HYDRO),
            aa_fraction(vl, HYDRO),
            net_charge(vh),
            net_charge(vl),
            len(vh),
            len(vl),
        ]

        # --- CDR lunghezze ---
        cdr_feats = [
            len(row.cdrh1),
            len(row.cdrh2),
            len(row.cdrh3),
            len(row.cdrl1),
            len(row.cdrl2),
            len(row.cdrl3),
        ]

        # --- embedding ESM: vh + vl concatenati ---
        emb = esm_embed_sequence(vh + vl)

        feats.append(seq_feats + cdr_feats + list(emb))

    return np.array(feats)


def train_adsentinel(train_df: pd.DataFrame, target: str) -> AdSentinelRegressor:
    """
    Addestra il modello AdSentinel su un target:
    es: 'hic', 'acsins', 'titer', 'tm2', ...
    """
    X = extract_all_features(train_df)
    y = train_df[target].values

    model = AdSentinelRegressor()
    model.fit(X, y)

    return model


def predict_adsentinel(model: AdSentinelRegressor, df: pd.DataFrame) -> np.ndarray:
    """
    Effettua la predizione usando lo stesso set di feature del training.
    """
    X = extract_all_features(df)
    return model.predict(X)
