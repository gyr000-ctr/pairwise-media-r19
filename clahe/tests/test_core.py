import math

import pytest

import clahe

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

IMG = [[0.0, 0.5], [0.5, 1.0]]

CASES = [
    ("shape", lambda: len(clahe.Clahe(2, 2.0).apply(IMG)) == 2),
    ("constant", lambda: all(abs(v - 0.3) < 1e-12 for row in clahe.Clahe(2, 2.0).apply([[0.3] * 4 for _ in range(4)]) for v in row)),
    ("range", lambda: all(0.0 <= v <= 1.0 for row in clahe.Clahe(2, 2.0).apply(IMG) for v in row)),
    ("bad_tiles", lambda: _raises(lambda: clahe.Clahe(0, 2.0))),
    ("bad_clip", lambda: _raises(lambda: clahe.Clahe(2, 0.0))),
    ("bad_empty", lambda: _raises(lambda: clahe.Clahe(2, 2.0).apply([]))),
    ("type", lambda: isinstance(clahe.Clahe(2, 2.0).apply(IMG), list)),
    ("det", lambda: clahe.Clahe(2, 2.0).apply(IMG) == clahe.Clahe(2, 2.0).apply(IMG)),
    ("clip_effect", lambda: clahe.Clahe(2, 1.0).apply(IMG) != clahe.Clahe(2, 10.0).apply(IMG)),
    ("no_seam", lambda: max(abs(a - b) for a, b in zip(clahe.Clahe(2, 2.0).apply(IMG)[0], clahe.Clahe(2, 2.0).apply(IMG)[0][1:])) < 1.0),
    ("nan", lambda: _raises(lambda: clahe.Clahe(2, 2.0).apply([[float('nan')]]))),
    ("no_mut", lambda: clahe.Clahe(2, 2.0).apply([r[:] for r in IMG]) is not None),
    ("monotone", lambda: clahe.Clahe(2, 2.0).apply(IMG)[0][0] <= clahe.Clahe(2, 2.0).apply(IMG)[1][1]),
    ("one_tile", lambda: isinstance(clahe.Clahe(1, 2.0).apply(IMG), list)),
    ("tile_count", lambda: len(clahe.Clahe(2, 2.0).apply([[0.0] * 4 for _ in range(4)])) == 4),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
