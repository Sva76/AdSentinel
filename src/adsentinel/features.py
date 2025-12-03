import numpy as np

HYDRO = set("AVLIMFWY")
POS = set("KRH")
NEG = set("DE")
AROM = set("FWYH")

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
