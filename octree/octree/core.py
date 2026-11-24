"""octree 核心 API（初始未实现）。"""


class Octree:
    def __init__(self, bounds, depth=8):
        """构造"""
        raise NotImplementedError

    def insert(self, name, p):
        """插入"""
        raise NotImplementedError

    def query(self, box):
        """范围查询"""
        raise NotImplementedError

    def nearest(self, p):
        """最近邻"""
        raise NotImplementedError
