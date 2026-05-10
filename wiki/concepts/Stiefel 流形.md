---
title: "Stiefel 流形"
type: concept
tags: [流形, 优化, 正交化, 数学基础]
sources: [raw/01-articles/zartbot-DeepSeek-V4详细分析（1）：算法和模型结构.md]
last_updated: 2026-05-10
---

## 定义
Stiefel 流形 V_k(R^n)（k ≤ n）是所有由 n 维空间中 k 个标准正交向量组成的有序集合构成的数学流形。在 Muon 优化器中，Newton-Schulz 正交化等价于将更新矩阵投影到 Stiefel 流形上，找到距离最近的"标准姿势"更新。

## 关键信息

### 正式定义
- V_k(R^n) = {X ∈ R^{n×k} | X^T X = I_k}
- 矩阵语言：X 是一个 n×k 矩阵，其列向量两两正交且范数为 1
- "标准正交" = orthogonal（正交）+ normal（单位长度）

### 与 Muon 的关系
- Muon 处理的是 m×n 权重矩阵，假设 m ≥ n
- 将动量更新 G 投影到 V_n(R^m) 上
- Newton-Schulz 迭代计算 G 的极分解的正交部分 U V^T（等价于 SVD 中的 U V^T）
- 验证：U V^T 满足条件 (U V^T)^T (U V^T) = V U^T U V^T = V V^T = I_n

### 几何直觉
- AdamW 在欧氏空间做梯度下降——各参数独立更新
- Muon 在 Stiefel 流形上运动——将"野路子"更新"拉回"到"名门正派"的轨道
- 刚性变换（旋转/反射）替代随意拉伸和扭曲

## 关联连接
- [[Muon Optimizer]] — 核心应用
- [[Newton-Schulz 迭代]] — 投影算法
- [[Birkhoff 多胞体]] — mHC 中的相关流形
