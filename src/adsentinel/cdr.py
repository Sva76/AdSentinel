def extract_cdrs(df):
    cdr_cols = [
        "cdrh1_aho", "cdrh2_aho", "cdrh3_aho",
        "cdrl1_aho", "cdrl2_aho", "cdrl3_aho"
    ]
    cdr_dict = {c: df[c].fillna("") for c in cdr_cols}
    return cdr_dict
