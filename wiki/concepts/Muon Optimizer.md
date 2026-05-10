---
title: "Muon Optimizer"
type: concept
tags: [优化器, 动量, 正交化, 流形优化, DeepSeek]
sources: [raw/01-articles/zartbot-DeepSeek-V4详细分析（1）：算法和模型结构.md]
last_updated: 2026-05-10
---

## 定义
Muon（MomentUm Orthogonalized by Newton-Schulz）是一种矩阵级优化器，专门针对神经网络隐藏层的二维参数更新。核心思想是将动量更新矩阵通过 Newton-Schulz 迭代正交化，投影到 Stiefel 流形上，用"刚性变换"替代逐元素更新。DeepSeek-V4 采用 Muon 作为主优化器替代 AdamW。

## 关键信息

### AdamW vs Muon
| 维度 | AdamW | Muon |
|------|-------|------|
| 操作粒度 | 逐元素 | 矩阵级 |
| 内存开销 | 2× 模型参数（m+v） | 1× 模型参数（仅 momentum） |
| 几何直觉 | 欧氏空间梯度下降 | Stiefel 流形上运动 |
| 更新原理 | 逐元素自适应步长 | 矩阵整体正交化后更新 |

### Newton-Schulz 迭代
核心是将矩阵 M = U Σ V^T 近似正交化为 U V^T（极分解的正交部分）：
- 初始归一化：X = G / (||G|| + ε)，确保奇异值在 [0,1]
- 迭代更新：X_{k+1} = a X_k + (b X_k X_k^T + c (X_k X_k^T)^2) X_k
- 转置保证始终计算较小方阵（m×m 或 n×n）

### DeepSeek 的混合 NS 迭代（共 10 轮）
- 前 8 步：系数 (3.4445, -4.7750, 2.0315)，激进收敛奇异值到 1
- 后 2 步：系数 (2, -1.5, 0.5)，精确稳定在 1

### Nesterov 加速 + 权重衰减 + Update RMS
- 使用 Nesterov 技巧：对"预见"方向做正交化
- 引入 AdamW 风格的权重衰减
- Update RMS 缩放：理论值为 √(min(m,n)/m)，实际缩放到 0.2-0.4 范围

### 适用范围
- ✅ 隐藏层二维权重（Linear 层）
- ✅ 卷积层四维参数（展平）
- ❌ Embedding / LM Head（逐 token 操作）
- ❌ 向量/标量参数（Bias、RMSNorm）
- DeepSeek-V4 中 Embedding、Prediction Head、RMSNorm、mHC 静态偏置仍保留 AdamW

### 计算效率
每轮 NS 迭代包含 3 次 GEMM + 逐元素操作，现代 GPU TensorCore 高效加速。BF16 精度执行。

## 关联连接
- [[DeepSeek-V4]] — 首次大规模应用
- [[Stiefel 流形]] — 理论基础
- [[Newton-Schulz 迭代]] — 核心算法
- [[Manifold-Constrained Hyper-Connections]] — 协同稳定训练
