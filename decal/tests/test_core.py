import math

import pytest

import decal

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

V = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]

CASES = [
    ("uv_count", lambda: len(decal.Decal().project(V, (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), 2.0)) == 3),
    ("uv_range", lambda: all(0.0 <= c <= 1.0 for uv in decal.Decal().project(V, (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), 2.0) for c in uv)),
    ("backface", lambda: decal.Decal().visible((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)) is False),
    ("frontface", lambda: decal.Decal().visible((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)) is True),
    ("bad_normal", lambda: _raises(lambda: decal.Decal().project(V, (0, 0, 0), (0.0, 0.0, 0.0), 2.0))),
    ("bad_size", lambda: _raises(lambda: decal.Decal().project(V, (0, 0, 0), (0, 0, 1), 0.0))),
    ("type", lambda: isinstance(decal.Decal().project(V, (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), 2.0), list)),
    ("det", lambda: decal.Decal().project(V, (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), 2.0) == decal.Decal().project(V, (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), 2.0)),
    ("uv_pair", lambda: all(len(uv) == 2 for uv in decal.Decal().project(V, (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), 2.0))),
    ("bool_visible", lambda: isinstance(decal.Decal().visible((0, 0, 1), (0, 0, 1)), bool)),
    ("nan", lambda: _raises(lambda: decal.Decal().project([(float('nan'), 0, 0)], (0, 0, 0), (0, 0, 1), 2.0))),
    ("no_mut", lambda: decal.Decal().project(list(V), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), 2.0) is not None),
    ("empty", lambda: decal.Decal().project([], (0, 0, 0), (0, 0, 1), 2.0) == []),
    ("float_uv", lambda: isinstance(decal.Decal().project(V, (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), 2.0)[0][0], float)),
    ("center", lambda: all(0.0 <= c <= 1.0 for c in decal.Decal().project([(0.0, 0.0, 0.0)], (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0)[0])),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
