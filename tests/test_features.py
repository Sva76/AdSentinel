import numpy as np

from adsentinel.features import aa_fraction, net_charge


def test_aa_fraction_basic():
    """
    Controlla che aa_fraction dia il valore atteso
    su una sequenza semplice.
    """
    seq = "ACDE"  # 4 amminoacidi, uno solo è 'A'
    frac_A = aa_fraction(seq, {"A"})
    # 1 su 4 = 0.25
    assert abs(frac_A - 0.25) < 1e-6


def test_aa_fraction_empty_returns_nan():
    """
    Sequenza vuota -> deve restituire NaN.
    """
    val = aa_fraction("", {"A"})
    assert np.isnan(val)


def test_net_charge_balanced_is_zero():
    """
    Sequenza con 1 positivo (K) e 1 negativo (E),
    lunghezza 3 -> (1 - 1)/3 = 0
    """
    seq = "KED"
    q = net_charge(seq)
    assert abs(q) < 1e-6


def test_net_charge_positive_sequence():
    """
    Solo residui positivi -> carica media > 0
    """
    seq = "KKKRR"
    q = net_charge(seq)
    assert q > 0
