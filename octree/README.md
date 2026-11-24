# octree

## 题面（可直接粘贴）

子目录：octree。八叉树空间索引。约束：范围查询与暴力一致；最近邻正确；插入重复点可区分报错；建树深度有上界；10 万点不超过 600 ms；并发查询安全。验收：cd octree && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd octree && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
