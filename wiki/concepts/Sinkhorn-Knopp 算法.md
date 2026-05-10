---
title: "Sinkhorn-Knopp 算法"
type: concept
tags: [算法, 矩阵归一化, 最优传输, DeepSeek]
sources: [raw/01-articles/zartbot-DeepSeek-V4详细分析（1）：算法和模型结构.md, raw/02-papers/mHC.pdf]
last_updated: 2026-05-10
---

## 定义
Sinkhorn-Knopp 算法是一种通过交替行归一化和列归一化将非负矩阵投影到双随机矩阵（每行和每列之和为 1）的迭代算法。在 DeepSeek-V4 的 mHC 中被用于将残差变换矩阵 B 投影到 Birkhoff 多胞体上，同时也是熵正则化最优传输问题的核心求解算法。

## 关键信息

### 算法核心（伪代码）
```python
for iter in range(sinkhorn_iters):
    comb = comb / row_sum(comb)    # 行和归一化
    comb = comb / col_sum(comb)    # 列和归一化
```

### 在 mHC 中的应用
1. 对原始参数 B 取 exp 确保为正
2. 迭代行归一化 + 列归一化（t_max=20）
3. 收敛到双随机矩阵，确保谱范数 ≤ 1

### 最优传输视角
Sinkhorn-Knopp 不只做投影，同时是熵正则化最优传输问题的求解器。暗示 mHC 的 B 矩阵可能在学习的是一种从上一层残差流到下一层残差流的"最优传输方案"：n_hc 个残差通道以最小成本将信息分配给下一层的 n_hc 个通道。

### 替代方案
mHC-Lite 讨论了不使用 Sinkhorn-Knopp 迭代的替代方案。但作者认为 20 次迭代开销可接受，工程上简单即可。

## 关联连接
- [[Manifold-Constrained Hyper-Connections]] — mHC 核心应用
- [[Birkhoff 多胞体]] — 双随机矩阵流形
- [[Stiefel 流形]] — 相关流形概念
