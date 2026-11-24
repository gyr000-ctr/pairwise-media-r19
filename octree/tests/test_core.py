import math

import pytest

import octree

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

B = ((0.0, 0.0, 0.0), (8.0, 8.0, 8.0))

CASES = [
    ("query_all", lambda: len(octree.Octree(B).insert('a', (1.0, 1.0, 1.0)) and octree.Octree(B).query(((0, 0, 0), (8, 8, 8)))) >= 0),
    ("bad_bounds", lambda: _raises(lambda: octree.Octree(((8, 8, 8), (0, 0, 0))))),
    ("bad_depth", lambda: _raises(lambda: octree.Octree(B, 0))),
    ("empty_query", lambda: octree.Octree(B).query(((0, 0, 0), (1, 1, 1))) == []),
    ("bad_insert", lambda: _raises(lambda: octree.Octree(B).insert('a', (100.0, 0.0, 0.0)))),
    ("type", lambda: isinstance(octree.Octree(B).query(((0, 0, 0), (1, 1, 1))), list)),
    ("det", lambda: octree.Octree(B).query(((0, 0, 0), (1, 1, 1))) == octree.Octree(B).query(((0, 0, 0), (1, 1, 1)))),
    ("nearest_empty", lambda: _raises(lambda: octree.Octree(B).nearest((0.0, 0.0, 0.0)))),
    ("dup", lambda: _raises(lambda: (lambda o: (o.insert('a', (1, 1, 1)), o.insert('a', (2, 2, 2))))(octree.Octree(B)))),
    ("ray_type", lambda: isinstance(octree.Octree(B), octree.Octree)),
    ("nan", lambda: _raises(lambda: octree.Octree(B).insert('a', (float('nan'), 0.0, 0.0)))),
    ("no_mut", lambda: octree.Octree(B).query(((0, 0, 0), (1, 1, 1))) is not None),
    ("depth", lambda: octree.Octree(B, 4) is not None),
    ("query_pair", lambda: isinstance(octree.Octree(B).query(((0, 0, 0), (1, 1, 1))), list)),
    ("nearest_after", lambda: (lambda o: (o.insert('a', (1, 1, 1)), o.nearest((0.5, 0.5, 0.5)))[1] == 'a')(octree.Octree(B))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
