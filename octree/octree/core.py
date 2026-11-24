"""八叉树空间索引。

设计要点：
- 节点只存两份并行表（首子下标、叶子桶链表头），节点中心不下发存储，
  而是在下降过程中由父中心与层级半边长递推，节省内存与写入开销；
  子节点按 8 个一组惰性批量创建，树的深度严格不超过构造参数 ``depth``。
- 范围查询基于节点 AABB 与查询盒求交，结果与逐点暴力过滤完全一致（闭区间），
  并按插入顺序返回。
- 最近邻用分支限界：按到查询点的最小距离排序子节点，不可更近的子树直接剪枝；
  等距时返回最早插入的点（剪枝条件为严格大于，不漏等距候选）。
- 重复点 / 重名插入抛出 ``ValueError``；非有限数（NaN/inf）及非法参数分类报错。
- 所有公开操作在同一把可重入锁下进行，并发查询（以及查询与插入并发）安全。
"""

import math
import threading

__all__ = ["Octree"]

# 建树深度的硬上界，防止误传超大 depth 造成资源耗尽。
MAX_DEPTH = 64

_EIGHT = (-1, -1, -1, -1, -1, -1, -1, -1)


def _vec3(v, what="point"):
    """把输入校验并转换成三元有限 float 组，非法输入抛 ValueError。"""
    if isinstance(v, (str, bytes, bytearray)):
        raise ValueError("%s must be a sequence of 3 numbers" % what)
    try:
        x, y, z = v
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be a sequence of 3 numbers" % what) from exc
    try:
        x = float(x)
        y = float(y)
        z = float(z)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s components must be numeric" % what) from exc
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        raise ValueError("%s components must be finite" % what)
    return x, y, z


class Octree:
    def __init__(self, bounds, depth=8):
        try:
            lo, hi = bounds
        except (TypeError, ValueError) as exc:
            raise ValueError("bounds must be ((x0,y0,z0),(x1,y1,z1))") from exc

        lx, ly, lz = _vec3(lo, "bounds min")
        hx, hy, hz = _vec3(hi, "bounds max")
        if not (lx < hx and ly < hy and lz < hz):
            raise ValueError("bounds min must be strictly less than max on every axis")

        if isinstance(depth, bool) or not isinstance(depth, int) or not (1 <= depth <= MAX_DEPTH):
            raise ValueError("depth must be an int in [1, %d]" % MAX_DEPTH)

        self._lo = (lx, ly, lz)
        self._hi = (hx, hy, hz)
        self._depth = depth
        self._root_center = ((lx + hx) * 0.5, (ly + hy) * 0.5, (lz + hz) * 0.5)

        # 各层级节点的半边长（根为第 0 层）。
        self._half = [
            (
                (hx - lx) / 2.0 ** level,
                (hy - ly) / 2.0 ** level,
                (hz - lz) / 2.0 ** level,
            )
            for level in range(depth + 1)
        ]

        # 节点表：_child[n] 为 n 的首个子节点下标（8 个连续），-1 未分裂；
        # _head[n] 为叶子桶链表头（项下标），-1 空桶。节点中心在遍历时递推。
        self._child = [-1]
        self._head = [-1]

        # 已插入点：名字列表、坐标列表、叶子桶内的链表 next 指针。
        self._names = []
        self._px = []
        self._py = []
        self._pz = []
        self._nxt = []

        self._name_set = set()
        self._point_set = set()

        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # 插入
    # ------------------------------------------------------------------ #
    def insert(self, name, p):
        """插入一个带名字的点，返回 self 以支持链式调用。"""
        x, y, z = _vec3(p, "point")
        lx, ly, lz = self._lo
        hx, hy, hz = self._hi
        if not (lx <= x <= hx and ly <= y <= hy and lz <= z <= hz):
            raise ValueError("point is outside the octree bounds")

        with self._lock:
            try:
                if name in self._name_set:
                    raise ValueError("duplicate point name: %r" % (name,))
            except TypeError as exc:
                raise ValueError("point name must be hashable") from exc
            key = (x, y, z)
            if key in self._point_set:
                raise ValueError("duplicate point at (%.17g, %.17g, %.17g)" % (x, y, z))

            idx = len(self._names)
            self._names.append(name)
            self._name_set.add(name)
            self._point_set.add(key)
            self._px.append(x)
            self._py.append(y)
            self._pz.append(z)

            child = self._child
            head = self._head
            half = self._half

            node = 0
            cx, cy, cz = self._root_center
            for level in range(self._depth):
                c0 = child[node]
                if c0 == -1:
                    # 惰性分裂：一次批量创建 8 个子节点。
                    c0 = len(head)
                    child[node] = c0
                    child.extend(_EIGHT)
                    head.extend(_EIGHT)

                shx, shy, shz = half[level + 1]
                # 位 0/1/2 分别对应 x/y/z，落在中心上归入高侧（保证 hi 边界可容纳）。
                octant = (
                    (1 if x >= cx else 0)
                    | (2 if y >= cy else 0)
                    | (4 if z >= cz else 0)
                )
                node = c0 + octant
                cx += shx if octant & 1 else -shx
                cy += shy if octant & 2 else -shy
                cz += shz if octant & 4 else -shz

            # 到达叶子（深度已达上界），挂入桶链表。
            self._nxt.append(head[node])
            head[node] = idx

        return self

    # ------------------------------------------------------------------ #
    # 范围查询
    # ------------------------------------------------------------------ #
    def query(self, box):
        """返回落在闭区间 box = (lo, hi) 内的所有点名（按插入顺序）。"""
        try:
            lo, hi = box
        except (TypeError, ValueError) as exc:
            raise ValueError("box must be ((x0,y0,z0),(x1,y1,z1))") from exc
        qx0, qy0, qz0 = _vec3(lo, "box min")
        qx1, qy1, qz1 = _vec3(hi, "box max")
        if not (qx0 <= qx1 and qy0 <= qy1 and qz0 <= qz1):
            raise ValueError("box min must not exceed max")

        with self._lock:
            if not self._names:
                return []

            child = self._child
            head = self._head
            half = self._half
            px = self._px
            py = self._py
            pz = self._pz
            nxt = self._nxt
            names = self._names
            depth = self._depth
            rcx, rcy, rcz = self._root_center

            hits = []
            # 栈帧：(节点, 层级, 中心 x/y/z)
            stack = [(0, 0, rcx, rcy, rcz)]
            while stack:
                node, level, cx, cy, cz = stack.pop()
                hhx, hhy, hhz = half[level]

                # 节点 AABB 与查询闭盒无交集则剪枝。
                if (
                    cx - hhx > qx1 or cx + hhx < qx0
                    or cy - hhy > qy1 or cy + hhy < qy0
                    or cz - hhz > qz1 or cz + hhz < qz0
                ):
                    continue

                if level == depth:
                    j = head[node]
                    while j != -1:
                        if (
                            qx0 <= px[j] <= qx1
                            and qy0 <= py[j] <= qy1
                            and qz0 <= pz[j] <= qz1
                        ):
                            hits.append((j, names[j]))
                        j = nxt[j]
                    continue

                c0 = child[node]
                if c0 == -1:
                    continue

                shx, shy, shz = half[level + 1]
                # 逆序压栈，保持 0..7 的确定性遍历顺序。
                for k in range(7, -1, -1):
                    ox = shx if k & 1 else -shx
                    oy = shy if k & 2 else -shy
                    oz = shz if k & 4 else -shz
                    stack.append((c0 + k, level + 1, cx + ox, cy + oy, cz + oz))

            hits.sort(key=lambda pair: pair[0])
            return [name for _, name in hits]

    # ------------------------------------------------------------------ #
    # 最近邻
    # ------------------------------------------------------------------ #
    def nearest(self, p):
        """返回离 p 最近的点名；等距时取最早插入者。空树抛 ValueError。"""
        x, y, z = _vec3(p, "point")

        with self._lock:
            if not self._names:
                raise ValueError("nearest() called on an empty octree")

            child = self._child
            head = self._head
            half = self._half
            px = self._px
            py = self._py
            pz = self._pz
            nxt = self._nxt
            names = self._names
            depth = self._depth
            rcx, rcy, rcz = self._root_center

            best_d = math.inf
            best_idx = -1
            best_name = None

            stack = [(0, 0, rcx, rcy, rcz)]
            while stack:
                node, level, cx, cy, cz = stack.pop()
                hhx, hhy, hhz = half[level]

                # p 到节点 AABB 的最小平方距离。
                dx = cx - hhx - x
                if dx < 0.0:
                    dx = x - (cx + hhx)
                    if dx < 0.0:
                        dx = 0.0
                dy = cy - hhy - y
                if dy < 0.0:
                    dy = y - (cy + hhy)
                    if dy < 0.0:
                        dy = 0.0
                dz = cz - hhz - z
                if dz < 0.0:
                    dz = z - (cz + hhz)
                    if dz < 0.0:
                        dz = 0.0
                min_d2 = dx * dx + dy * dy + dz * dz

                # 等于当前最优也不能剪：等距点可能插入更早，需要参与比较。
                if best_idx != -1 and min_d2 > best_d:
                    continue

                if level == depth:
                    j = head[node]
                    while j != -1:
                        ddx = px[j] - x
                        ddy = py[j] - y
                        ddz = pz[j] - z
                        d2 = ddx * ddx + ddy * ddy + ddz * ddz
                        if d2 < best_d or (d2 == best_d and j < best_idx):
                            best_d = d2
                            best_idx = j
                            best_name = names[j]
                        j = nxt[j]
                    continue

                c0 = child[node]
                if c0 == -1:
                    continue

                # 子节点按到 p 的最小距离升序访问，尽早收紧上界。
                shx, shy, shz = half[level + 1]
                scored = []
                for k in range(8):
                    ccx = cx + (shx if k & 1 else -shx)
                    ccy = cy + (shy if k & 2 else -shy)
                    ccz = cz + (shz if k & 4 else -shz)
                    ex = ccx - shx - x
                    if ex < 0.0:
                        ex = x - (ccx + shx)
                        if ex < 0.0:
                            ex = 0.0
                    ey = ccy - shy - y
                    if ey < 0.0:
                        ey = y - (ccy + shy)
                        if ey < 0.0:
                            ey = 0.0
                    ez = ccz - shz - z
                    if ez < 0.0:
                        ez = z - (ccz + shz)
                        if ez < 0.0:
                            ez = 0.0
                    scored.append((ex * ex + ey * ey + ez * ez, k))
                scored.sort()
                for _, k in reversed(scored):
                    ox = shx if k & 1 else -shx
                    oy = shy if k & 2 else -shy
                    oz = shz if k & 4 else -shz
                    stack.append((c0 + k, level + 1, cx + ox, cy + oy, cz + oz))

            return best_name

    def __len__(self):
        with self._lock:
            return len(self._names)
