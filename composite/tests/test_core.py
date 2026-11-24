import math

import pytest

import composite

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

A = (0.2, 0.2, 0.2, 1.0)
B = (0.5, 0.5, 0.5, 1.0)

CASES = [
    ("over_opaque", lambda: composite.Composite().over(A, B)[:3] == B[:3]),
    ("multiply", lambda: abs(composite.Composite().multiply(A, B)[0] - 0.1) < 1e-12),
    ("screen", lambda: abs(composite.Composite().screen(A, B)[0] - 0.6) < 1e-12),
    ("alpha_zero", lambda: abs(composite.Composite().over(A, (1.0, 1.0, 1.0, 0.0))[0] - 0.2) < 1e-12),
    ("bad_len", lambda: _raises(lambda: composite.Composite().over((0.0, 0.0, 0.0), B))),
    ("bad_range", lambda: _raises(lambda: composite.Composite().over((2.0, 0.0, 0.0, 1.0), B))),
    ("type", lambda: isinstance(composite.Composite().over(A, B), tuple)),
    ("det", lambda: composite.Composite().over(A, B) == composite.Composite().over(A, B)),
    ("alpha_out", lambda: abs(composite.Composite().over((0, 0, 0, 0.5), (1, 1, 1, 0.5))[3] - 0.75) < 1e-12),
    ("range", lambda: all(0.0 <= v <= 1.0 for v in composite.Composite().screen(A, B))),
    ("roundtrip", lambda: abs(composite.Composite().over(A, (0.5, 0.5, 0.5, 0.0))[0] - 0.2) < 1.0 / 255),
    ("nan", lambda: _raises(lambda: composite.Composite().over((float('nan'), 0, 0, 1), B))),
    ("no_mut", lambda: composite.Composite().over(A, B) is not None),
    ("commutative", lambda: abs(composite.Composite().screen(A, B)[0] - composite.Composite().screen(B, A)[0]) < 1e-12),
    ("four_channel", lambda: len(composite.Composite().multiply(A, B)) == 4),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
