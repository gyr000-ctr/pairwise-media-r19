"""decal 核心 API：贴花正交投影。

设计要点：
- ``Decal`` 实例无状态，所有计算只使用局部变量，天然并发安全；
- 不修改调用方传入的顶点序列；
- 错误分类（均为 ``ValueError`` 子类，可互相区分）：
  - ``InvalidNormalError``：法线为零向量；
  - ``InvalidSizeError``：投影尺寸非正；
  - ``NonFiniteError``：任何输入含 NaN/Inf 或不可转为浮点；
- 正交基采用手工构造：up 取 (0,1,0)（法线接近平行时回退 (1,0,0)），
  tangent = normalize(cross(up, n))，bitangent = cross(n, tangent)。
"""

import math

__all__ = [
    "Decal",
    "DecalError",
    "InvalidNormalError",
    "InvalidSizeError",
    "NonFiniteError",
]


class DecalError(ValueError):
    """贴花参数错误基类。"""


class InvalidNormalError(DecalError):
    """法线为零向量，无法构造投影基。"""


class InvalidSizeError(DecalError):
    """投影尺寸非法（非正数）。"""


class NonFiniteError(DecalError):
    """输入包含 NaN、无穷或不可转为浮点的分量。"""


def _as_vec3(value, name):
    """把输入解释为有限三分量浮点向量，否则抛出可分类错误。"""
    try:
        x, y, z = value
    except (TypeError, ValueError) as exc:
        raise DecalError("%s 必须是三分量向量" % name) from exc
    try:
        fx, fy, fz = float(x), float(y), float(z)
    except (TypeError, ValueError) as exc:
        raise NonFiniteError("%s 含非数值分量" % name) from exc
    if not (math.isfinite(fx) and math.isfinite(fy) and math.isfinite(fz)):
        raise NonFiniteError("%s 含非有限分量 (NaN/Inf)" % name)
    return fx, fy, fz


def _normalize_normal(normal):
    nx, ny, nz = _as_vec3(normal, "normal")
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length == 0.0:
        raise InvalidNormalError("法线为零向量，无法投影")
    return nx / length, ny / length, nz / length


def _orthonormal_basis(n):
    """由单位法线构造手工正交基 (tangent, bitangent)。"""
    nx, ny, nz = n
    if abs(ny) < 0.999:
        ux, uy, uz = 0.0, 1.0, 0.0
    else:
        ux, uy, uz = 1.0, 0.0, 0.0
    # tangent = normalize(cross(up, n))
    tx = uy * nz - uz * ny
    ty = uz * nx - ux * nz
    tz = ux * ny - uy * nx
    tl = math.sqrt(tx * tx + ty * ty + tz * tz)
    tx, ty, tz = tx / tl, ty / tl, tz / tl
    # bitangent = cross(n, tangent)（两者皆单位且正交，结果仍为单位向量）
    bx = ny * tz - nz * ty
    by = nz * tx - nx * tz
    bz = nx * ty - ny * tx
    return (tx, ty, tz), (bx, by, bz)


def _clamp01(value):
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


class Decal:
    """贴花投影器。无状态，可安全地在多线程间共享。"""

    def basis(self, normal):
        """返回 (tangent, bitangent, normal) 手工正交基（均为单位向量）。"""
        n = _normalize_normal(normal)
        tangent, bitangent = _orthonormal_basis(n)
        return tangent, bitangent, n

    def projection_matrix(self, origin, normal, size):
        """返回 4x4 行主序投影矩阵，将世界坐标映射到 (u, v, depth, 1)。"""
        n = _normalize_normal(normal)
        ox, oy, oz = _as_vec3(origin, "origin")
        inv = 1.0 / self._validate_size(size)
        (tx, ty, tz), (bx, by, bz) = _orthonormal_basis(n)
        nx, ny, nz = n
        return [
            [tx * inv, ty * inv, tz * inv, 0.5 - (ox * tx + oy * ty + oz * tz) * inv],
            [bx * inv, by * inv, bz * inv, 0.5 - (ox * bx + oy * by + oz * bz) * inv],
            [nx, ny, nz, -(ox * nx + oy * ny + oz * nz)],
            [0.0, 0.0, 0.0, 1.0],
        ]

    @staticmethod
    def _validate_size(size):
        try:
            s = float(size)
        except (TypeError, ValueError) as exc:
            raise InvalidSizeError("投影尺寸必须是数值") from exc
        if not math.isfinite(s):
            raise NonFiniteError("投影尺寸非有限 (NaN/Inf)")
        if s <= 0.0:
            raise InvalidSizeError("投影尺寸必须为正数")
        return s

    def project(self, verts, origin, normal, size):
        """把顶点投影到贴花平面，返回 [[u, v], ...]，UV 钳制在 [0, 1]。"""
        n = _normalize_normal(normal)
        ox, oy, oz = _as_vec3(origin, "origin")
        inv = 1.0 / self._validate_size(size)
        (tx, ty, tz), (bx, by, bz) = _orthonormal_basis(n)

        result = []
        append = result.append
        for vertex in verts:
            vx, vy, vz = _as_vec3(vertex, "vertex")
            dx = vx - ox
            dy = vy - oy
            dz = vz - oz
            u = 0.5 + (dx * tx + dy * ty + dz * tz) * inv
            v = 0.5 + (dx * bx + dy * by + dz * bz) * inv
            append([_clamp01(u), _clamp01(v)])
        return result

    def visible(self, normal, face_normal):
        """背向面剔除：贴花法线与面法线点积为正时可见。"""
        nx, ny, nz = _normalize_normal(normal)
        fx, fy, fz = _normalize_normal(face_normal)
        return (nx * fx + ny * fy + nz * fz) > 0.0
