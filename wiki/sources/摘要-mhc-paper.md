---
title: "摘要-mhc-paper"
type: source
tags: [论文, 残差连接, 流形约束, 深度训练]
sources: [raw/02-papers/mHC.pdf]
last_updated: 2026-05-10
---

## 核心摘要
mHC（Manifold-Constrained Hyper-Connections）论文是 DeepSeek-V4 中残差连接设计的理论基石。该论文提出了将残差变换矩阵约束到双随机矩阵流形（Birkhoff 多胞体）上的方法，通过 Sinkhorn-Knopp 算法实现投影，确保残差映射的非扩张性（谱范数 ≤ 1），从而在深层堆叠时维持前向/反向传播的数值稳定性。相比无约束的 Hyper-Connections，mHC 解决了深层训练时的损失尖峰问题。

## 关联连接
- [[Manifold-Constrained Hyper-Connections]] — mHC 概念页
- [[DeepSeek-V4]] — 将 mHC 应用于生产级 LLM
- [[Sinkhorn-Knopp 算法]] — 核心投影算法
- [[Birkhoff 多胞体]] — 双随机矩阵流形
