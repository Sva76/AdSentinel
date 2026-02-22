import pandas as pd

def extract_cdrs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns dataframe enriched with standardized CDR columns.
    Expected input columns:
    cdrh1_aho, cdrh2_aho, cdrh3_aho,
    cdrl1_aho, cdrl2_aho, cdrl3_aho
    """

    df = df.copy()

    mapping = {
        "cdrh1_aho": "cdrh1",
        "cdrh2_aho": "cdrh2",
        "cdrh3_aho": "cdrh3",
        "cdrl1_aho": "cdrl1",
        "cdrl2_aho": "cdrl2",
        "cdrl3_aho": "cdrl3",
    }

    for old, new in mapping.items():
        if old in df.columns:
            df[new] = df[old].fillna("")
        else:
            df[new] = ""

    return df
