---
title: "Lightning Indexer"
type: concept
tags: [注意力机制, 稀疏选择, 索引器, DeepSeek]
sources: [raw/01-articles/zartbot-DeepSeek-V4详细分析（1）：算法和模型结构.md, raw/01-articles/DeepSeek-V4技术报告解读-从架构到 Infra 的全栈重构.md]
last_updated: 2026-05-10
---

## 定义
Lightning Indexer 是 DeepSeek-V4 CSA 注意力中的稀疏选择组件，负责在压缩 KV 条目中快速选出 top-k 个与当前查询最相关的条目。采用低秩、低精度设计以确保选择过程本身不成为瓶颈。

## 关键信息

### 核心设计
- 查询下投影得到压缩潜向量 c_Q（维度远小于 hidden_dim）
- 再上投影出多头 indexer query
- Indexer 的 QK 计算使用 **FP4 精度**加速
- Indexer 与主注意力**共享潜查询向量 c_Q**，避免重复投影

### 计算流程
1. `wq_b` 投影生成 indexer query（与主 Q 共享 c_Q）
2. `weights_proj` 计算多头权重
3. Q 做 Hadamard 旋转 + FP4 量化
4. K 使用共享的 indexer K cache
5. FP4 精度计算 index score
6. Top-k selector 选出最高分的压缩块

### 固定算力开销（V4-Pro）
- wq_b 投影 + weights_proj：26.1M FLOPs/token
- Indexer 压缩器：7.3M FLOPs/token
- Indexer Score 计算：4096 × Seq FLOPs

### 效率优化
- Index score 从 FP32 量化到 BF16，top-k selector 提速 2×
- KV recall 仍保持 99.7%

## 关联连接
- [[Compressed Sparse Attention]] — CSA 注意力
- [[FP4 量化感知训练]] — FP4 量化技术
- [[DeepSeek-V4]] — 主体模型
