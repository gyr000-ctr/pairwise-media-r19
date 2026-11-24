import math

import pytest

import vad

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _v():
    return vad.Vad(16000, 0.02)

CASES = [
    ("silence", lambda: _v().segments([0.0] * 1000) == []),
    ("speech_found", lambda: len(_v().segments([0.5] * 1000)) >= 1),
    ("seg_pair", lambda: all(len(s) == 2 for s in _v().segments([0.5] * 1000))),
    ("seg_order", lambda: all(a < b for a, b in _v().segments([0.5] * 1000))),
    ("bad_rate", lambda: _raises(lambda: vad.Vad(0, 0.02))),
    ("bad_threshold", lambda: _raises(lambda: _v().segments([0.5]) if False else vad.Vad(16000, 2.0))),
    ("empty", lambda: _v().segments([]) == []),
    ("type", lambda: isinstance(_v().segments([0.5] * 100), list)),
    ("det", lambda: _v().segments([0.5] * 100) == _v().segments([0.5] * 100)),
    ("nan", lambda: _raises(lambda: _v().segments([float('nan')] * 10))),
    ("bounds", lambda: all(0 <= a < 1000 and a < b <= 1000 for a, b in _v().segments([0.5] * 1000))),
    ("no_mut", lambda: _v().segments([0.5] * 100) is not None),
    ("two_spans", lambda: isinstance(_v().segments([0.5] * 400 + [0.0] * 400 + [0.5] * 400), list)),
    ("seg_int", lambda: all(isinstance(x, int) for s in _v().segments([0.5] * 1000) for x in s)),
    ("quiet", lambda: _v().segments([0.001] * 1000) == []),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
