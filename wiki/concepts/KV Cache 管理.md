---
title: "KV Cache 管理"
type: concept
tags: [推理, 缓存, 长上下文, 内存管理]
sources: [raw/01-articles/DeepSeek-V4技术报告解读-从架构到 Infra 的全栈重构.md, raw/01-articles/zartbot-DeepSeek-V4详细分析（1）：算法和模型结构.md]
last_updated: 2026-05-10
---

## 定义
KV Cache 管理是大模型推理中的关键技术，涉及如何高效存储、检索和复用注意力机制的键值对缓存。DeepSeek-V4 通过混合注意力架构（CSA+HCA）和分级缓存设计实现了百万上下文下极致的 KV Cache 效率。

## 关键信息

### DeepSeek-V4 KV Cache 结构
**State Cache**（每请求固定大小）：
- SWA 的 n_win token
- CSA/HCA 不足压缩块的尾部 token

**Classical KV Cache**（按 lcm(m, m')=128 对齐分块）：
- CSA 压缩 KV（ratio=4，FP8 精度）
- HCA 压缩 KV（ratio=128，FP8 精度）
- CSA Indexer KV（FP4 精度）

### 混合精度存储
- RoPE 维度：BF16
- 其余维度：FP8（相比纯 BF16 减少近一半 KV Cache）

### On-Disk Prefix Reuse
三种 SWA 缓存策略：
- **Full SWA Caching**：全存，无重算但写放大严重
- **Periodic Checkpointing**：每 p token checkpoint，按需加载+部分重算
- **Zero SWA Caching**：不存 SWA，利用 CSA/HCA KV 还原

### 效率对比（1M context vs V3.2）
- V4-Pro：KV Cache 仅为 V3.2 的 10%
- V4-Flash：KV Cache 仅为 V3.2 的 7%
- 对比 BF16 GQA8 基线：仅为约 2%

### V3.2 KV Cache 对比
| 组件 | V3.2 格式 | V4-Pro 等效 |
|------|-----------|-------------|
| kv_cache | FP8, Seq×61×512 | CSA/HCA 压缩 KV |
| pe_cache | BF16, Seq×61×64 | 混合精度 |
| indexer_k_cache | FP8, Seq×61×128 | FP4 |

## 关联连接
- [[Compressed Sparse Attention]] — CSA 压缩策略
- [[Heavily Compressed Attention]] — HCA 压缩策略
- [[DeepSeek-V4]] — 主体模型
- [[FP4 量化感知训练]] — 量化技术
