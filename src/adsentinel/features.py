import numpy as np
from Bio import SeqIO
from typing import Dict
import torch
import esm

# ---------------------------
# Amino acid sets
# ---------------------------
HYDRO = set("AVLIMFWY")
POS = set("KRH")
NEG = set("DE")
AROM = set("FWYH")

# ---------------------------
# Basic residue statistics
# ---------------------------
def aa_fraction(seq, aa_set):
    seq = "".join([a for a in seq if a.isalpha()])
    if not seq:
        return np.nan
    return sum(a in aa_set for a in seq) / len(seq)

def net_charge(seq):
    seq = "".join([a for a in seq if a.isalpha()])
    if not seq:
        return np.nan
    pos = sum(a in POS for a in seq)
    neg = sum(a in NEG for a in seq)
    return (pos - neg) / len(seq)

# ---------------------------
# CDR extraction (AHo)
# ---------------------------
def extract_cdrs(aho_seq: str) -> Dict[str, str]:
    """
    Extracts CDR1, CDR2, CDR3 using AHo numbering.
    Expects input with gaps removed.
    """
    # Approximate AHo positions for demonstration
    return {
        "cdr1": aho_seq[26:38],
        "cdr2": aho_seq[55:65],
        "cdr3": aho_seq[105:135]  # handles variable-length CDR-H3
    }

def gravy(seq):
    KD = {
        'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,
        'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,
        'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3
    }
    seq = "".join([a for a in seq if a.isalpha()])
    if not seq:
        return np.nan
    return sum(KD.get(a, 0) for a in seq) / len(seq)

def shannon_entropy(seq):
    seq = "".join([a for a in seq if a.isalpha()])
    if not seq:
        return np.nan
    from collections import Counter
    L = len(seq)
    counts = Counter(seq)
    return -sum((c/L)*np.log2(c/L) for c in counts.values())

# ---------------------------
# ESM-2 Embeddings
# ---------------------------
print("Loading ESM-2 model…")
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
esm_model.eval()

def esm_embedding(vh, vl):
    seq = vh + vl
    batch = [( "antibody", seq )]
    batch_labels, batch_strs, batch_tokens = batch_converter(batch)

    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33])
        token_reps = results["representations"][33]

    seq_rep = token_reps[0, 1:len(seq)+1]
    return seq_rep.mean(0).numpy()

# ---------------------------
# Full Feature Extractor
# ---------------------------
def compute_adsentinel_features(vh, vl, vh_aho=None, vl_aho=None):
    feats = {}

    # Basic sequence features
    feats["vh_len"] = len(vh)
    feats["vl_len"] = len(vl)
    feats["vh_hydro"] = aa_fraction(vh, HYDRO)
    feats["vl_hydro"] = aa_fraction(vl, HYDRO)
    feats["vh_charge"] = net_charge(vh)
    feats["vl_charge"] = net_charge(vl)
    feats["vh_arom"] = aa_fraction(vh, AROM)
    feats["vl_arom"] = aa_fraction(vl, AROM)

    # CDR features (if AHo provided)
    if vh_aho:
        cdrs = extract_cdrs(vh_aho)
        for name, seq in cdrs.items():
            feats[f"{name}_len"] = len(seq)
            feats[f"{name}_gravy"] = gravy(seq)
            feats[f"{name}_entropy"] = shannon_entropy(seq)

    # ESM embedding
    emb = esm_embedding(vh, vl)
    for i, v in enumerate(emb):
        feats[f"esm_{i}"] = v

    return feats
