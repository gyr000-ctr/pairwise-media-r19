"""decal 核心 API（初始未实现）。"""


class Decal:
    def project(self, verts, origin, normal, size):
        """投影 UV"""
        raise NotImplementedError

    def visible(self, normal, face_normal):
        """是否可见"""
        raise NotImplementedError
