# decal

## 题面（可直接粘贴）

子目录：decal。贴花投影。约束：投影到平面后 UV 在 [0,1]；背向面剔除；投影矩阵与手工正交基一致（≤1e-9）；非法法线可区分报错；1 万顶点不超过 300 ms；并发安全。验收：cd decal && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd decal && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
