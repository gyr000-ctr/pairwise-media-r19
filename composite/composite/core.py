"""composite 核心 API（初始未实现）。"""


class Composite:
    def over(self, dst, src):
        """source-over"""
        raise NotImplementedError

    def multiply(self, dst, src):
        """multiply"""
        raise NotImplementedError

    def screen(self, dst, src):
        """screen"""
        raise NotImplementedError
