"""vad 核心 API（初始未实现）。"""


class Vad:
    def __init__(self, rate, threshold):
        """构造"""
        raise NotImplementedError

    def segments(self, samples):
        """语音段"""
        raise NotImplementedError
