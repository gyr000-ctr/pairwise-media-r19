# composite

## 题面（可直接粘贴）

子目录：composite。多层预乘合成。约束：over/multiply/screen 与 W3C 公式逐通道一致（≤1e-12）；alpha=0 层不改变结果；层序反转会改变结果（除不透明外）；预乘反预乘往返 ≤1/255；非法层可区分报错；100 万像素不超过 500 ms。验收：cd composite && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd composite && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
