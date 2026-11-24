# clahe

## 题面（可直接粘贴）

子目录：clahe。CLAHE（限制对比度自适应均衡）。约束：块边界双线性融合无接缝（相邻差有上界）；cliplevel 限制对比度（超限裁剪并重分配）；块大小可配且结果与逐块计算一致；常量图不变（≤1e-12）；100 万像素不超过 800 ms；非法参数可区分报错。验收：cd clahe && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd clahe && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
