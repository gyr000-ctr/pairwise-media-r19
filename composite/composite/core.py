"""W3C Compositing and Blending Level 1 实现。

公开 API：

* ``Composite`` —— 无状态标量合成器，每个像素一个 RGBA 四元组，返回四元组。
* ``blend_pixels`` —— 向量化逐像素合成，接受 ``(..., 4)`` 的 numpy 数组。
* ``composite_layers`` —— 多图层叠（支持隔离 / 非隔离组语义）。
* ``premultiply`` / ``unpremultiply`` —— 预乘 / 反预乘工具。

颜色一律使用 straight（非预乘）RGBA、float64、取值 [0, 1]。
混合公式见 https://www.w3.org/TR/compositing-1/#blending ，
组合成公式见 https://www.w3.org/TR/compositing-1/#simplealphacompositing 。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "Composite",
    "blend_pixels",
    "composite_layers",
    "premultiply",
    "unpremultiply",
    "BLEND_MODES",
]

# W3C 规定的全部 16 种混合模式（normal 即 source-over 的混合函数 B=Cs）。
BLEND_MODES = (
    "normal",
    "multiply",
    "screen",
    "overlay",
    "darken",
    "lighten",
    "color-dodge",
    "color-burn",
    "hard-light",
    "soft-light",
    "difference",
    "exclusion",
    "hue",
    "saturation",
    "color",
    "luminosity",
)

# 常见别名 -> 规范名。
_ALIASES = {
    "over": "normal",
    "source-over": "normal",
    "color_dodge": "color-dodge",
    "color_burn": "color-burn",
    "hard_light": "hard-light",
    "soft_light": "soft-light",
    "dodge": "color-dodge",
    "burn": "color-burn",
}

_LUM_W = np.array([0.3, 0.59, 0.11], dtype=np.float64)


# --------------------------------------------------------------------------- #
# 输入校验
# --------------------------------------------------------------------------- #
def _canonical_mode(mode: str) -> str:
    if not isinstance(mode, str):
        raise ValueError(f"mode 必须是字符串，收到 {type(mode).__name__}")
    key = mode.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in BLEND_MODES:
        raise ValueError(f"未知混合模式: {mode!r}")
    return key


def _as_rgba(value, name: str):
    """把输入转成 float64 的 (..., 4) 数组，并做完整合法性校验。

    返回 (array, is_scalar)。
    """
    if isinstance(value, np.ndarray):
        arr = value
        if arr.dtype.kind == "b":
            # bool 数组拒绝（语义模糊）。
            raise ValueError(f"{name} 必须是数值数组")
        if arr.dtype.kind not in ("f", "i", "u"):
            raise ValueError(f"{name} 必须是数值数组")
        if arr.shape == ():
            raise ValueError(f"{name} 必须是 4 通道 RGBA")
        # uint8 是 8 位图像的标准载体，按 /255 归一化；其余整数按 float 语义
        # 视为 [0,1]（范围检查会拦截越界）。
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float64) / 255.0
        else:
            arr = arr.astype(np.float64, copy=False)
        is_scalar = False
    else:
        try:
            seq = list(value)
        except TypeError as exc:
            raise ValueError(f"{name} 必须是 4 通道 RGBA 序列") from exc
        if len(seq) != 4:
            raise ValueError(f"{name} 必须恰好包含 4 个通道，收到 {len(seq)} 个")
        arr = np.asarray(seq, dtype=np.float64)
        is_scalar = True

    if arr.shape[-1] != 4 or arr.ndim < 1:
        raise ValueError(f"{name} 的最后一维必须是 4 个 RGBA 通道")

    # 非有限值（NaN / inf）必须分类报错，且先于范围检查。
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} 包含非有限值（NaN 或 inf）")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"{name} 的通道值必须落在 [0, 1] 内")
    return arr, is_scalar


# --------------------------------------------------------------------------- #
# 可分离混合模式（输入 straight 颜色，形状 (..., 3)）
# --------------------------------------------------------------------------- #
def _soft_light_d(x):
    """W3C soft-light 的 D(x) 分段函数。"""
    branch_poly = ((16.0 * x - 12.0) * x + 4.0) * x
    branch_sqrt = np.sqrt(x)
    return np.where(x <= 0.25, branch_poly, branch_sqrt)


def _blend_separable(cb, cs, mode):
    if mode == "normal":
        return cs
    if mode == "multiply":
        return cb * cs
    if mode == "screen":
        return cb + cs - cb * cs
    if mode == "darken":
        return np.minimum(cb, cs)
    if mode == "lighten":
        return np.maximum(cb, cs)
    if mode == "difference":
        return np.abs(cb - cs)
    if mode == "exclusion":
        return cb + cs - 2.0 * cb * cs
    if mode == "overlay":
        # overlay(B, C) = hard-light(C, B)，分段依据是 backdrop Cb。
        lo = 2.0 * cs * cb
        hi = 1.0 - 2.0 * (1.0 - cs) * (1.0 - cb)
        return np.where(cb <= 0.5, lo, hi)
    if mode == "hard-light":
        lo = 2.0 * cs * cb
        hi = 1.0 - 2.0 * (1.0 - cs) * (1.0 - cb)
        return np.where(cs <= 0.5, lo, hi)
    if mode == "soft-light":
        d = _soft_light_d(cb)
        lo = cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb)
        hi = cb + (2.0 * cs - 1.0) * (d - cb)
        return np.where(cs <= 0.5, lo, hi)
    if mode == "color-dodge":
        # W3C 判定顺序：Cb==0 先返回 0；否则 Cs==1 返回 1；再走除法。
        # 分母为 0 的位置换成 1，避免除零产生 inf（后续 0*inf 会变 NaN）。
        denom = np.where(cs < 1.0, 1.0 - cs, 1.0)
        q = np.minimum(1.0, cb / denom)
        return np.where(cb <= 0.0, 0.0, np.where(cs >= 1.0, 1.0, q))
    if mode == "color-burn":
        # W3C 判定顺序：Cb==1 先返回 1；否则 Cs==0 返回 0；再走除法。
        denom = np.where(cs > 0.0, cs, 1.0)
        q = 1.0 - np.minimum(1.0, (1.0 - cb) / denom)
        return np.where(cb >= 1.0, 1.0, np.where(cs <= 0.0, 0.0, q))
    raise AssertionError("not separable")  # pragma: no cover


# --------------------------------------------------------------------------- #
# 不可分离混合模式
# --------------------------------------------------------------------------- #
def _lum(c):
    return c @ _LUM_W


def _clip_color(c):
    """W3C ClipColor：先压负分量，再压超过 1 的分量（顺序敏感）。"""
    l = _lum(c)[..., None]
    mn = c.min(axis=-1, keepdims=True)
    mx = c.max(axis=-1, keepdims=True)

    denom_lo = np.where(mn < 0.0, l - mn, 1.0)
    c_lo = l + (c - l) * l / denom_lo
    c = np.where(mn < 0.0, c_lo, c)

    mx2 = c.max(axis=-1, keepdims=True)
    denom_hi = np.where(mx2 > 1.0, mx2 - l, 1.0)
    c_hi = l + (c - l) * (1.0 - l) / denom_hi
    c = np.where(mx2 > 1.0, c_hi, c)
    return c


def _set_lum(c, l):
    return _clip_color(c + (l[..., None] - _lum(c)[..., None]))


def _sat(c):
    return c.max(axis=-1) - c.min(axis=-1)


def _set_sat(c, s):
    """按 W3C SetSat：最小值处置 0，最大值处置 s，中间值按比例缩放。"""
    order = np.argsort(c, axis=-1)
    lo = np.take_along_axis(c, order[..., 0:1], axis=-1)
    mid = np.take_along_axis(c, order[..., 1:2], axis=-1)
    hi = np.take_along_axis(c, order[..., 2:3], axis=-1)

    span = hi - lo
    mid_new = np.where(span > 0.0, (mid - lo) * s[..., None] / np.where(span > 0.0, span, 1.0), 0.0)
    out = np.empty_like(c)
    np.put_along_axis(out, order[..., 0:1], np.zeros_like(lo), axis=-1)
    np.put_along_axis(out, order[..., 1:2], mid_new, axis=-1)
    np.put_along_axis(out, order[..., 2:3], s[..., None], axis=-1)
    return out


def _blend_nonseparable(cb, cs, mode):
    if mode == "hue":
        return _set_lum(_set_sat(cs, _sat(cb)), _lum(cb))
    if mode == "saturation":
        # W3C: SetLum(SetSat(Cs, Sat(Cb)), Lum(Cb))
        return _set_lum(_set_sat(cs, _sat(cb)), _lum(cb))
    if mode == "color":
        return _set_lum(cs, _lum(cb))
    if mode == "luminosity":
        return _set_lum(cb, _lum(cs))
    raise AssertionError("not non-separable")  # pragma: no cover


_NONSEPARABLE = frozenset(("hue", "saturation", "color", "luminosity"))


def _blend_straight(cb, cs, mode):
    """对 straight RGB 颜色执行 W3C 混合函数 B(Cb, Cs)。"""
    if mode in _NONSEPARABLE:
        return _blend_nonseparable(cb, cs, mode)
    return _blend_separable(cb, cs, mode)


# --------------------------------------------------------------------------- #
# 核心合成（W3C §9 简单 alpha 合成，混合在 straight 颜色上进行）
# --------------------------------------------------------------------------- #
def _composite_validated(dst, src, mode):
    """合成内核。调用方负责保证 dst/src 已是合法 float64 RGBA 数组。"""
    cb = dst[..., :3]
    cs = src[..., :3]
    ab = dst[..., 3:4]
    as_ = src[..., 3:4]

    blended = _blend_straight(cb, cs, mode)

    # co = αs(1-αb)Cs + αsαb B(Cb,Cs) + (1-αs)αb Cb
    co = (
        as_ * (1.0 - ab) * cs
        + as_ * ab * blended
        + (1.0 - as_) * ab * cb
    )
    ao = as_ + ab - as_ * ab
    # αo=0 时 co 必为 0；安全选择分母，避免 0/0 告警。
    color = np.where(ao > 0.0, co / np.where(ao > 0.0, ao, 1.0), 0.0)
    out = np.concatenate([color, ao], axis=-1)
    # 源完全透明的像素整像素短路：结果与 dst 逐位一致
    #（除法 (αb·Cb)/αb 会引入 1-ULP 误差，无法仅在颜色层抵消）。
    zero_src = np.broadcast_to(as_ <= 0.0, out.shape)
    out = np.where(zero_src, dst, out)
    # 消掉浮点毛刺并保证合法范围。
    out = np.clip(out, 0.0, 1.0)
    return out


def blend_pixels(dst, src, mode="normal"):
    """向量化逐像素合成。

    ``dst`` / ``src`` 为 shape ``(..., 4)`` 的数值数组或单个 RGBA 序列。
    数组输入返回 ndarray；序列输入返回 tuple。
    """
    mode = _canonical_mode(mode)
    dst_arr, d_scalar = _as_rgba(dst, "dst")
    src_arr, s_scalar = _as_rgba(src, "src")
    if dst_arr.shape != src_arr.shape:
        raise ValueError("dst 与 src 形状必须一致")
    out = _composite_validated(dst_arr, src_arr, mode)
    if d_scalar and s_scalar:
        return tuple(float(v) for v in out)
    return out


# --------------------------------------------------------------------------- #
# 多图层 / 组隔离
# --------------------------------------------------------------------------- #
def _layer_stack(arrays, mode, isolated, background):
    """在数组层面折叠图层栈。"""
    if background is None:
        # 隔离组的初始底衬是透明黑；非隔离组未给父底衬时等价。
        acc = np.zeros_like(arrays[0])
    else:
        acc, _ = _as_rgba(background, "background")
        if acc.shape != arrays[0].shape:
            raise ValueError("background 与图层形状不一致")

    if isolated:
        # 隔离组：组内以透明黑为底衬逐层混合，最后整体 source-over 到父底衬。
        group = np.zeros_like(arrays[0])
        for layer in arrays:
            group = _composite_validated(group, layer, mode)
        if background is not None:
            acc = _composite_validated(acc, group, "normal")
        else:
            acc = group
    else:
        # 非隔离组：父底衬是组内隐式的最底层元素，参与逐层折叠。
        for layer in arrays:
            acc = _composite_validated(acc, layer, mode)
    return acc


def composite_layers(layers, mode="normal", *, isolated=True, background=None):
    """把多个 RGBA 图层自底向上合成。

    Parameters
    ----------
    layers:
        RGBA 序列的列表（返回 tuple），或 shape ``(N, ..., 4)`` 的数组（返回 ndarray）。
    mode:
        任意 :data:`BLEND_MODES` 名称（支持下划线/连字符与 over 别名）。
    isolated:
        True 时按 W3C 隔离组处理（组内底衬为透明黑，结果再 source-over 到
        ``background``）；False 时 ``background`` 作为父底衬直接参与混合。
    background:
        父底衬 RGBA；默认透明黑。
    """
    mode = _canonical_mode(mode)
    if not isinstance(layers, (list, tuple, np.ndarray)) or len(layers) == 0:
        raise ValueError("layers 必须是非空的图层序列")

    if isinstance(layers, np.ndarray):
        if layers.ndim < 2 or layers.shape[-1] != 4:
            raise ValueError("数组图层的形状必须是 (N, ..., 4)")
        arrays = [layers[i] for i in range(layers.shape[0])]
        result = _layer_stack(arrays, mode, isolated, background)
        return result

    scalar_layers = []
    for i, layer in enumerate(layers):
        arr, is_scalar = _as_rgba(layer, f"layers[{i}]")
        scalar_layers.append((arr, is_scalar))
    if not all(s for _, s in scalar_layers):
        raise ValueError("图层输入形式必须一致：全部为标量序列或全部为数组")
    arrays = [a for a, _ in scalar_layers]
    result = _layer_stack(arrays, mode, isolated, background)
    return tuple(float(v) for v in result)


# --------------------------------------------------------------------------- #
# 预乘 / 反预乘
# --------------------------------------------------------------------------- #
def premultiply(rgba, dtype=np.float64):
    """straight RGBA -> 预乘 RGBA。

    ``dtype=np.uint8`` 时输出 8 位量化数组（颜色 floor，alpha round）。
    """
    arr, is_scalar = _as_rgba(rgba, "rgba")
    alpha = arr[..., 3:4]
    pre = arr.copy()
    pre[..., :3] = arr[..., :3] * alpha
    if dtype is np.uint8:
        # 颜色分量向下取整：配合反预乘的四舍五入，8 位网格输入的 straight 域
        # 往返严格不超过 1/255（两次 round 会累积到 2/255，故颜色不能用 round）；
        # alpha 通道本身用四舍五入。
        q_color = np.floor(pre[..., :3] * 255.0 + 1e-9).clip(0, 255).astype(np.uint8)
        q_alpha = np.rint(pre[..., 3:4] * 255.0).clip(0, 255).astype(np.uint8)
        q = np.concatenate([q_color, q_alpha], axis=-1)
        if is_scalar:
            return tuple(int(v) for v in q)
        return q
    out = pre.astype(np.float64, copy=False)
    if is_scalar:
        return tuple(float(v) for v in out)
    return out


def unpremultiply(prgba, dtype=np.float64):
    """预乘 RGBA -> straight RGBA。alpha=0 的像素反预乘为 0。"""
    if isinstance(prgba, np.ndarray) and prgba.dtype == np.uint8:
        arr = prgba.astype(np.float64) / 255.0
        is_scalar = False
    elif isinstance(prgba, tuple) and len(prgba) == 4 and all(
        isinstance(v, (int, np.integer)) for v in prgba
    ):
        arr = np.asarray(prgba, dtype=np.float64) / 255.0
        is_scalar = True
    else:
        arr, is_scalar = _as_rgba(prgba, "prgba")
    alpha = arr[..., 3:4]
    out = arr.copy()
    out[..., :3] = np.where(alpha > 0.0, arr[..., :3] / np.where(alpha > 0.0, alpha, 1.0), 0.0)
    out = np.clip(out, 0.0, 1.0)
    if dtype is np.uint8:
        # 反预乘四舍五入；与 premultiply 的向下取整配合，
        # 8 位网格输入的 straight 域往返严格 <= 1/255。
        q = np.rint(out * 255.0).clip(0, 255).astype(np.uint8)
        if is_scalar:
            return tuple(int(v) for v in q)
        return q
    if is_scalar:
        return tuple(float(v) for v in out)
    return out


# --------------------------------------------------------------------------- #
# 标量门面（保持题面 API：Composite().over(dst, src) -> 4-tuple）
# --------------------------------------------------------------------------- #
class Composite:
    """无状态标量合成器；线程安全，可任意并发复用。"""

    # 每个公开模式一个标量方法，统一走向量化内核（单像素时开销可忽略）。
    def _call(self, dst, src, mode):
        return blend_pixels(dst, src, mode)

    def blend(self, dst, src, mode="normal"):
        return self._call(dst, src, _canonical_mode(mode))

    def normal(self, dst, src):
        return self._call(dst, src, "normal")

    def over(self, dst, src):
        return self._call(dst, src, "normal")

    def multiply(self, dst, src):
        return self._call(dst, src, "multiply")

    def screen(self, dst, src):
        return self._call(dst, src, "screen")

    def overlay(self, dst, src):
        return self._call(dst, src, "overlay")

    def darken(self, dst, src):
        return self._call(dst, src, "darken")

    def lighten(self, dst, src):
        return self._call(dst, src, "lighten")

    def color_dodge(self, dst, src):
        return self._call(dst, src, "color-dodge")

    def color_burn(self, dst, src):
        return self._call(dst, src, "color-burn")

    def hard_light(self, dst, src):
        return self._call(dst, src, "hard-light")

    def soft_light(self, dst, src):
        return self._call(dst, src, "soft-light")

    def difference(self, dst, src):
        return self._call(dst, src, "difference")

    def exclusion(self, dst, src):
        return self._call(dst, src, "exclusion")

    def hue(self, dst, src):
        return self._call(dst, src, "hue")

    def saturation(self, dst, src):
        return self._call(dst, src, "saturation")

    def color(self, dst, src):
        return self._call(dst, src, "color")

    def luminosity(self, dst, src):
        return self._call(dst, src, "luminosity")
