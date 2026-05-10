---
title: "Multi-Head Latent Attention"
type: concept
tags: [注意力机制, KV压缩, 低秩, DeepSeek]
sources: [raw/01-articles/DeepSeekV4中RoPE设计解析.md, raw/01-articles/zartbot-DeepSeek-V4详细分析（1）：算法和模型结构.md]
last_updated: 2026-05-10
---

## 定义
多头潜在注意力（Multi-Head Latent Attention, MLA）是 DeepSeek-V2/V3/V3.2 使用的注意力机制，通过 KV 低秩压缩减少 KV Cache 显存占用。K 与 V 共享低秩表示（仅需存储一份压缩 KV），同时通过部分 RoPE 设计解决位置编码污染 V 值的问题。

## 关键信息

### 核心设计
- **KV 低秩压缩**：原始 KV 维度 d → 低秩维度 kv_lora_rank（V3.2 为 512）
- **K/V 共享**：K 和 V 共享同一份压缩表示，节省 KV Cache 显存
- **部分 RoPE**：在 Q 和 K 的隐藏维度中设置一部分专门用于 RoPE 计算（qk_rope_head_dim=64），避免位置编码污染 V 值

### 与 V4 CSA/HCA 的关系
MLA 是 V4 混合注意力的前身：
- V3.2：MLA + DSA（DeepSeek Sparse Attention）
- V4：CSA + HCA（压缩 + 稀疏/密集混合）

V4 的 CSA/HCA 也继承了 MLA 的部分 RoPE 设计理念。

## 关联连接
- [[Compressed Sparse Attention]] — V4 的 CSA
- [[Heavily Compressed Attention]] — V4 的 HCA
- [[DeepSeek-V3.2]] — 使用 MLA 的模型
- [[DeepSeek-V4]] — 下一代模型
- [[RoPE]] — 旋转位置编码
