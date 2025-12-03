def extract_all_features(df):
    """
    Estrae tutte le feature di AdSentinel:
    - feature globali VH/VL
    - feature CDR (lunghezze, ecc.)
    - embedding ESM
    """
    df = df.copy()
    # Arricchisco il dataframe UNA volta con le colonne CDR
    # Supponiamo che extract_cdrs(df) aggiunga colonne:
    # cdrh1, cdrh2, cdrh3, cdrl1, cdrl2, cdrl3
    df = extract_cdrs(df)

    feats = []

    for row in df.itertuples(index=False):
        vh = row.vh_protein_sequence
        vl = row.vl_protein_sequence

        # --- feature globali VH/VL ---
        seq_feats = [
            aa_fraction(vh, HYDRO),
            aa_fraction(vl, HYDRO),
            net_charge(vh),
            net_charge(vl),
            len(vh),
            len(vl),
        ]

        # --- feature CDR (esempio: solo lunghezze) ---
        cdr_seqs = [
            row.cdrh1,
            row.cdrh2,
            row.cdrh3,
            row.cdrl1,
            row.cdrl2,
            row.cdrl3,
        ]
        cdr_feats = [len(c) if isinstance(c, str) else 0 for c in cdr_seqs]

        # --- embedding ESM (vh+vl concatenati, o come preferisci) ---
        emb = esm_embed_sequence(vh + vl)  # deve restituire un vettore 1D

        feats.append(seq_feats + cdr_feats + list(emb))

    return np.array(feats)
