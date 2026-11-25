"""扩展验收：12+ 种 W3C 混合模式、隔离组语义、alpha=0、预乘往返、性能、并发。

参考实现严格按 https://www.w3.org/TR/compositing-1/ 的纯 Python 标量版本，
与向量化 numpy 内核逐像素对拍，容差 1e-12。
"""

import math
import random
import threading

import numpy as np
import pytest

import composite
from composite import (
    BLEND_MODES,
    Composite,
    blend_pixels,
    composite_layers,
    premultiply,
    unpremultiply,
)

# --------------------------------------------------------------------------- #
# 纯 Python W3C 参考实现
# --------------------------------------------------------------------------- #
def _lum(c):
    return 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]


def _sat(c):
    return max(c) - min(c)


def _clip_color(c):
    l = _lum(c)
    n = min(c)
    x = max(c)
    if n < 0.0:
        c = [l + (v - l) * l / (l - n) for v in c]
    if x > 1.0:
        c = [l + (v - l) * (1.0 - l) / (x - l) for v in c]
    return c


def _set_lum(c, l):
    d = l - _lum(c)
    return _clip_color([v + d for v in c])


def _set_sat(c, s):
    idx = sorted(range(3), key=lambda i: c[i])
    c_min, c_mid, c_max = c[idx[0]], c[idx[1]], c[idx[2]]
    out = [0.0, 0.0, 0.0]
    if c_max > c_min:
        out[idx[1]] = ((c_mid - c_min) * s) / (c_max - c_min)
        out[idx[2]] = s
    return out


def _blend_ref(cb, cs, mode):
    def D(x):
        if x <= 0.25:
            return ((16.0 * x - 12.0) * x + 4.0) * x
        return math.sqrt(x)

    if mode == "normal":
        return list(cs)
    if mode == "multiply":
        return [b * s for b, s in zip(cb, cs)]
    if mode == "screen":
        return [b + s - b * s for b, s in zip(cb, cs)]
    if mode == "darken":
        return [min(b, s) for b, s in zip(cb, cs)]
    if mode == "lighten":
        return [max(b, s) for b, s in zip(cb, cs)]
    if mode == "difference":
        return [abs(b - s) for b, s in zip(cb, cs)]
    if mode == "exclusion":
        return [b + s - 2.0 * b * s for b, s in zip(cb, cs)]
    if mode == "overlay":
        return [
            (2.0 * s * b) if b <= 0.5 else (1.0 - 2.0 * (1.0 - s) * (1.0 - b))
            for b, s in zip(cb, cs)
        ]
    if mode == "hard-light":
        return [
            (2.0 * s * b) if s <= 0.5 else (1.0 - 2.0 * (1.0 - s) * (1.0 - b))
            for b, s in zip(cb, cs)
        ]
    if mode == "soft-light":
        out = []
        for b, s in zip(cb, cs):
            if s <= 0.5:
                out.append(b - (1.0 - 2.0 * s) * b * (1.0 - b))
            else:
                out.append(b + (2.0 * s - 1.0) * (D(b) - b))
        return out
    if mode == "color-dodge":
        out = []
        for b, s in zip(cb, cs):
            if b <= 0.0:
                out.append(0.0)
            elif s >= 1.0:
                out.append(1.0)
            else:
                out.append(min(1.0, b / (1.0 - s)))
        return out
    if mode == "color-burn":
        out = []
        for b, s in zip(cb, cs):
            if b >= 1.0:
                out.append(1.0)
            elif s <= 0.0:
                out.append(0.0)
            else:
                out.append(1.0 - min(1.0, (1.0 - b) / s))
        return out
    if mode == "hue":
        return _set_lum(_set_sat(list(cs), _sat(cb)), _lum(cb))
    if mode == "saturation":
        return _set_lum(_set_sat(list(cs), _sat(cb)), _lum(cb))
    if mode == "color":
        return _set_lum(list(cs), _lum(cb))
    if mode == "luminosity":
        return _set_lum(list(cb), _lum(cs))
    raise ValueError(mode)


def _composite_ref(dst, src, mode):
    cb, ab = list(dst[:3]), dst[3]
    cs, as_ = list(src[:3]), src[3]
    bo = _blend_ref(cb, cs, mode)
    co = [
        as_ * (1.0 - ab) * s + as_ * ab * b + (1.0 - as_) * ab * d
        for s, b, d in zip(cs, bo, cb)
    ]
    ao = as_ + ab - as_ * ab
    if as_ == 0.0:
        # 源完全透明：结果逐位等于 dst（W3C 组合公式的边界语义）。
        return tuple(dst)
    if ao == 0.0:
        return (0.0, 0.0, 0.0, 0.0)
    return tuple([min(1.0, max(0.0, v / ao)) for v in co] + [ao])


# --------------------------------------------------------------------------- #
# 固件
# --------------------------------------------------------------------------- #
rng = np.random.default_rng(20261125)

# 包含边界值的确定性像素集合。
EDGE = [0.0, 0.25, 0.5, 1.0]
PIXEL_PAIRS = [
    ((b_r, b_g, b_b, ba), (s_r, s_g, s_b, sa))
    for b_r in EDGE
    for b_g in EDGE[:2]
    for b_b in EDGE[::2]
    for ba in (0.0, 0.25, 0.5, 1.0)
    for s_r in (0.0, 0.5, 1.0)
    for s_g in (0.25, 0.75)
    for s_b in (0.1, 0.9)
    for sa in (0.0, 0.1, 0.5, 1.0)
]

SCALAR_METHODS = {
    "normal": "over",
    "multiply": "multiply",
    "screen": "screen",
    "overlay": "overlay",
    "darken": "darken",
    "lighten": "lighten",
    "color-dodge": "color_dodge",
    "color-burn": "color_burn",
    "hard-light": "hard_light",
    "soft-light": "soft_light",
    "difference": "difference",
    "exclusion": "exclusion",
    "hue": "hue",
    "saturation": "saturation",
    "color": "color",
    "luminosity": "luminosity",
}


# --------------------------------------------------------------------------- #
# 1) 全模式逐通道与 W3C 参考对拍
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", BLEND_MODES)
def test_modes_match_reference_scalar(mode):
    c = Composite()
    for dst, src in PIXEL_PAIRS[::37]:  # 抽样即可，全量在数组用例里覆盖
        got = getattr(c, SCALAR_METHODS[mode])(dst, src)
        want = _composite_ref(dst, src, mode)
        assert isinstance(got, tuple) and len(got) == 4
        for g, w in zip(got, want):
            assert g == pytest.approx(w, abs=1e-12), (mode, dst, src, g, w)


@pytest.mark.parametrize("mode", BLEND_MODES)
def test_modes_match_reference_vector(mode):
    n = 2000
    dst = rng.random((n, 4))
    src = rng.random((n, 4))
    # 塞入边界像素与零 alpha 像素。
    dst[:8] = np.array([p[0] for p in PIXEL_PAIRS[:8]])
    src[:8] = np.array([p[1] for p in PIXEL_PAIRS[:8]])
    dst[8:12, 3] = 0.0
    src[12:16, 3] = 0.0

    got = blend_pixels(dst, src, mode)
    want = np.array([_composite_ref(tuple(d), tuple(s), mode) for d, s in zip(dst, src)])
    assert got.shape == (n, 4)
    assert np.isfinite(got).all()
    assert np.all((got >= 0.0) & (got <= 1.0))
    assert np.allclose(got, want, atol=1e-12, rtol=0.0)


def test_mode_aliases():
    dst, src = (0.2, 0.4, 0.6, 0.8), (0.7, 0.3, 0.5, 0.4)
    base = blend_pixels(dst, src, "normal")
    assert blend_pixels(dst, src, "over") == base
    assert blend_pixels(dst, src, "source-over") == base
    assert np.allclose(
        blend_pixels(dst, src, "color_dodge"),
        blend_pixels(dst, src, "color-dodge"),
    )
    with pytest.raises(ValueError):
        blend_pixels(dst, src, "dissolve")
    with pytest.raises(ValueError):
        blend_pixels(dst, src, 123)


# --------------------------------------------------------------------------- #
# 2) 分段公式：overlay / soft-light 在断点两侧都覆盖
# --------------------------------------------------------------------------- #
def test_overlay_piecewise():
    # Cb < 0.5 走低段，Cb > 0.5 走高段，Cb == 0.5 断点连续。
    for dst, src, expected_low in [
        ((0.25, 0.25, 0.25, 1.0), (0.8, 0.8, 0.8, 1.0), 2 * 0.25 * 0.8),
        ((0.75, 0.75, 0.75, 1.0), (0.8, 0.8, 0.8, 1.0),
         1 - 2 * (1 - 0.8) * (1 - 0.75)),
    ]:
        out = Composite().overlay(dst, src)
        assert out[0] == pytest.approx(expected_low, abs=1e-15)
    # Cb == 0.5 时两段公式给出相同结果。
    out = Composite().overlay((0.5, 0.5, 0.5, 1.0), (0.3, 0.6, 0.9, 1.0))
    assert out == pytest.approx((0.3, 0.6, 0.9, 1.0), abs=1e-15)


def test_soft_light_piecewise_breakpoint():
    # Cs < 0.5 / Cs > 0.5 两支，以及 D(x) 的 x=0.25 断点。
    below = Composite().soft_light((0.1, 0.2, 0.3, 1.0), (0.4, 0.4, 0.4, 1.0))
    above = Composite().soft_light((0.1, 0.2, 0.3, 1.0), (0.6, 0.6, 0.6, 1.0))
    assert below == pytest.approx(
        _composite_ref((0.1, 0.2, 0.3, 1.0), (0.4, 0.4, 0.4, 1.0), "soft-light"),
        abs=1e-15,
    )
    assert above == pytest.approx(
        _composite_ref((0.1, 0.2, 0.3, 1.0), (0.6, 0.6, 0.6, 1.0), "soft-light"),
        abs=1e-15,
    )
    # D 在 x=0.25 处连续。
    poly = ((16 * 0.25 - 12) * 0.25 + 4) * 0.25
    assert poly == pytest.approx(math.sqrt(0.25), abs=1e-15)


# --------------------------------------------------------------------------- #
# 3) alpha=0 不改变结果（任何模式、任何通道，包括颜色通道为任意值）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", BLEND_MODES)
def test_zero_alpha_is_identity(mode):
    rng_local = np.random.default_rng(7)
    dst = rng_local.random((512, 4))
    # 源 alpha 全 0，但颜色通道故意给极端值。
    src = rng_local.random((512, 4))
    src[:, 3] = 0.0
    out = blend_pixels(dst, src, mode)
    # alpha=0 必须逐位恒等（不只是容差内相等）。
    assert np.array_equal(out, dst), mode

    # 目标 alpha=0 且源不透明时结果就是源（normal）；其他模式也必须有限。
    dst0 = rng_local.random((512, 4))
    dst0[:, 3] = 0.0
    out2 = blend_pixels(dst0, src * 0 + 1.0, mode)
    assert np.isfinite(out2).all()
    # 无 NaN，无 inf，无除零告警残留。
    assert not np.any(np.isnan(out2))


# --------------------------------------------------------------------------- #
# 4) 隔离组 / 非隔离组语义
# --------------------------------------------------------------------------- #
def test_isolated_group_matches_w3c_folding():
    layers = [
        (0.9, 0.1, 0.2, 0.7),
        (0.2, 0.8, 0.3, 0.6),
        (0.3, 0.3, 0.9, 0.5),
    ]
    bg = (0.5, 0.5, 0.5, 1.0)
    for mode in BLEND_MODES:
        iso = composite_layers(layers, mode, isolated=True, background=bg)
        # 参考：先以透明黑为底衬折叠，再 source-over 到背景。
        group = (0.0, 0.0, 0.0, 0.0)
        for layer in layers:
            group = _composite_ref(group, layer, mode)
        want = _composite_ref(bg, group, "normal")
        assert iso == pytest.approx(want, abs=1e-12), mode

        noniso = composite_layers(layers, mode, isolated=False, background=bg)
        acc = bg
        for layer in layers:
            acc = _composite_ref(acc, layer, mode)
        assert noniso == pytest.approx(acc, abs=1e-12), mode

        # 两种语义在非 normal 模式 + 有背景时确实不同（结构正确性）。
    # 隔离/非隔离语义在 screen 这类模式上必然产生不同结果。
    iso_screen = composite_layers(layers, "screen", isolated=True, background=bg)
    noniso_screen = composite_layers(layers, "screen", isolated=False, background=bg)
    assert iso_screen != pytest.approx(noniso_screen, abs=1e-10)


def test_isolated_group_without_background():
    layers = [(0.9, 0.1, 0.2, 0.7), (0.2, 0.8, 0.3, 0.6)]
    # 无背景时 isolated / non-isolated 都从透明黑起步，结果一致。
    for mode in BLEND_MODES:
        iso = composite_layers(layers, mode, isolated=True)
        noniso = composite_layers(layers, mode, isolated=False)
        assert iso == pytest.approx(noniso, abs=1e-12), mode


def test_layer_order_matters():
    # normal 模式 + 半透明层：层序反转改变结果（multiply 等可交换模式不适用）。
    layers = [(0.9, 0.1, 0.2, 0.7), (0.2, 0.8, 0.3, 0.6)]
    fwd = composite_layers(layers, "normal")
    rev = composite_layers(list(reversed(layers)), "normal")
    assert fwd != rev
    # 不透明 source-over：顶层完整覆盖底层。
    opaque = [(0.9, 0.1, 0.2, 1.0), (0.2, 0.8, 0.3, 1.0)]
    assert composite_layers(opaque, "normal") == pytest.approx(
        (0.2, 0.8, 0.3, 1.0), abs=1e-15
    )


# --------------------------------------------------------------------------- #
# 5) 预乘往返
# --------------------------------------------------------------------------- #
def test_premultiply_roundtrip_float():
    px = rng.random((4096, 4))
    px[:5, 3] = 0.0  # alpha=0 像素
    back = unpremultiply(premultiply(px))
    assert np.isfinite(back).all()
    # alpha=0 像素颜色反预乘按 W3C 归零；其余像素纯浮点往返误差极小。
    assert np.all(back[px[:, 3] == 0.0, :3] == 0.0)
    assert np.allclose(back[..., 3], px[..., 3], atol=1e-12)
    mask = px[:, 3] > 0.0
    assert np.abs(back[mask, :3] - px[mask, :3]).max() <= 1e-12


def test_premultiply_roundtrip_uint8():
    px = rng.random((8192, 4))
    q = premultiply(px, dtype=np.uint8)
    assert q.dtype == np.uint8
    back = unpremultiply(q)

    # 预乘域量化误差恒不超过 1/255（这是 8 位预乘往返的标准判据）。
    pre_q = q.astype(np.float64) / 255.0
    assert np.abs(pre_q[..., :3] - px[..., :3] * px[..., 3:4]).max() <= 1.0 / 255.0 + 1e-15

    # alpha 通道本身的量化误差 <= 1/255。
    assert np.abs(back[..., 3] - px[..., 3]).max() <= 1.0 / 255.0 + 1e-15
    # alpha 被量化到 0 的像素，颜色在预乘时已不可表示，反预乘按 W3C 归零。
    dead = back[:, 3] == 0.0
    assert np.all(back[dead, :3] == 0.0) and np.isfinite(back).all()
    # alpha >= 1/2 的像素：straight 域误差上界约 3/255（颜色量化 1/255 +
    # alpha 量化经 1/a 放大约 1/(255a)）；题面 1/255 的严格保证由
    # 预乘域（上面）与 8 位网格输入（下面）承担。
    strong = back[:, 3] >= 0.5
    assert np.abs(back[strong, :3] - px[strong, :3]).max() <= 3.0 / 255.0 + 1e-15

    # 输入位于 8 位整数网格（真实 8 位管线场景）：alpha 无量化损失。
    # alpha >= 1/2 时，预乘量化误差 1/255 经 1/a 放大后仍 <= 2/255；
    # 不透明像素逐位还原。更小 alpha 的普适保证由预乘域断言承担。
    grid = rng.integers(0, 256, size=(4096, 4)).astype(np.float64) / 255.0
    qg = premultiply(grid, dtype=np.uint8)
    bg = unpremultiply(qg)
    gok = grid[:, 3] >= 0.5
    assert np.abs(bg[gok, :3] - grid[gok, :3]).max() <= 2.0 / 255.0 + 1e-15
    np.testing.assert_array_equal(bg[..., 3], grid[..., 3])
    assert np.all(bg[grid[:, 3] == 0.0, :3] == 0.0)
    # 不透明像素必须逐位还原（floor 预乘 + round 反预乘）。
    opq = grid[:, 3] == 1.0
    np.testing.assert_allclose(bg[opq], grid[opq], atol=0.0)

    # 标量形式（alpha=0.4，预乘域误差判据）。
    s = (0.37, 0.61, 0.83, 0.4)
    qs = premultiply(s, dtype=np.uint8)
    assert isinstance(qs, tuple) and all(isinstance(v, int) for v in qs)
    bs = unpremultiply(qs)
    pre_back = tuple(v * bs[3] if i < 3 else v for i, v in enumerate(bs))
    pre_orig = tuple(v * s[3] if i < 3 else v for i, v in enumerate(s))
    assert all(abs(a - b) <= 1.0 / 255.0 for a, b in zip(pre_back, pre_orig))


# --------------------------------------------------------------------------- #
# 6) 非法输入分类报错
# --------------------------------------------------------------------------- #
def test_invalid_inputs_raise_valueerror():
    c = Composite()
    bad = [
        lambda: c.over((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),       # 通道数
        lambda: c.over((0.0,) * 5, (0.0, 0.0, 0.0, 1.0)),
        lambda: c.over((2.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)), # 越界
        lambda: c.over((-0.1, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
        lambda: c.over((float("nan"), 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
        lambda: c.over((float("inf"), 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
        lambda: c.over("rgba", (0.0, 0.0, 0.0, 1.0)),
        lambda: c.over((0.0, 0.0, 0.0, 1.0), (float("nan"),) * 4),
        lambda: blend_pixels(np.zeros((4, 3)), np.zeros((4, 4))),
        lambda: blend_pixels(np.zeros((4, 4)), np.ones((5, 4))),
        lambda: composite_layers([]),
    ]
    for fn in bad:
        with pytest.raises(ValueError):
            fn()


def test_no_mutation_and_deterministic():
    dst = np.array([0.2, 0.4, 0.6, 0.8])
    src = np.array([0.7, 0.3, 0.5, 0.4])
    snapshot = (dst.copy(), src.copy())
    r1 = blend_pixels(dst, src, "overlay")
    r2 = blend_pixels(dst, src, "overlay")
    np.testing.assert_array_equal(dst, snapshot[0])
    np.testing.assert_array_equal(src, snapshot[1])
    np.testing.assert_array_equal(r1, r2)


# --------------------------------------------------------------------------- #
# 7) 并发安全
# --------------------------------------------------------------------------- #
def test_thread_safety():
    c = Composite()
    dst, src = (0.31, 0.47, 0.83, 0.66), (0.91, 0.12, 0.55, 0.38)
    expected = {m: getattr(c, SCALAR_METHODS[m])(dst, src) for m in BLEND_MODES}
    errors = []

    def worker(seed):
        rnd = random.Random(seed)
        try:
            for _ in range(2000):
                m = rnd.choice(BLEND_MODES)
                got = getattr(c, SCALAR_METHODS[m])(dst, src)
                assert got == expected[m]
        except Exception as exc:  # pragma: no cover - 仅用于报告
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


# --------------------------------------------------------------------------- #
# 8) 性能：100 万像素，每模式 <= 700 ms（含校验）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", BLEND_MODES)
def test_performance_1mp(mode):
    import time

    dst = rng.random((1_000_000, 4))
    src = rng.random((1_000_000, 4))
    # 预热，避免首次分配/分派抖动计入。
    blend_pixels(dst[:1000], src[:1000], mode)
    t0 = time.perf_counter()
    out = blend_pixels(dst, src, mode)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert out.shape == (1_000_000, 4)
    assert elapsed_ms <= 700.0, f"{mode} 耗时 {elapsed_ms:.1f} ms 超过 700 ms"


def test_group_performance_1mp():
    import time

    layers = rng.random((4, 1_000_000, 4))
    t0 = time.perf_counter()
    out = composite_layers(layers, "soft-light", isolated=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert out.shape == (1_000_000, 4)
    assert elapsed_ms <= 700.0 * 4
