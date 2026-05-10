---
title: "RoPE"
type: concept
tags: [位置编码, 注意力机制, Transformer]
sources: [raw/01-articles/DeepSeekV4中RoPE设计解析.md]
last_updated: 2026-05-10
---

## 定义
旋转位置编码（Rotary Position Embedding, RoPE）是一种通过旋转变换将绝对位置信息注入 Q/K 向量的位置编码方法。其核心性质是 Q 和 K 的内积仅依赖相对位置（n-m），具有良好的长上下文外推能力。

## 关键信息

### 核心公式
- Q_m^T K_n = (R_m q_m)^T (R_n k_n) = q_m^T R_{n-m} k_n
- R(θ)^T = R(θ)^{-1} = R(-θ)（正交性）

### DeepSeek MLA 中的部分 RoPE
- 在 Q、K 隐藏维度中设专门 RoPE 维度（如 rope_head_dim=64）
- 避免 K 与 V 共享表示时位置信息污染 V 值
- 只需额外存储较小的 RoPE K cache（k_pe），远小于完整拆分 K/V

### V4 CSA/HCA 中的 RoPE 设计
- **压缩后旋转**（非压缩前）：避免位置信息在序列维度累加混合
- **HCA 取段起始位置**作为旋转角度参考
- **对输出 O 做逆旋转**：将绝对位置信息转为相对位置，保证长上下文可扩展性
- **窗口通道 + 压缩通道均施加 RoPE**

## 关联连接
- [[Multi-Head Latent Attention]] — MLA 中的 RoPE
- [[Compressed Sparse Attention]] — CSA 中的 RoPE
- [[Heavily Compressed Attention]] — HCA 中的 RoPE
- [[DeepSeek-V4]] — V4 中的位置编码设计
