---
title: "DeepSeek-V3.2"
type: entity
tags: [DeepSeek, LLM, MoE, 稀疏注意力]
sources: [raw/02-papers/DeepSeek_V3_2.pdf, raw/01-articles/zartbot-DeepSeek-V4详细分析（1）：算法和模型结构.md]
last_updated: 2026-05-10
---

## 定义
DeepSeek-V3.2 是 DeepSeek-V4 的直接前代模型，采用 DeepSeekMoE 架构 + DSA（DeepSeek Sparse Attention） + MTP（Multi-Token Prediction）设计。V3.2 在 V4 技术报告中被作为效率对比的基线模型。

## 关键信息

### 核心架构
- 模型维度 d = 7168
- 注意力头数 128，Q 低秩维度 1536，KV 低秩维度 512
- 总层数 61（3 层稠密 MLP + 58 层 MoE）
- 激活专家数 8，共享专家 1
- 索引头数 64，索引 TopK 2048

### MLA 注意力
V3.2 使用 Multi-Head Latent Attention（MLA），通过 KV 低秩压缩实现 KV Cache 缩减。采用 DSA（DeepSeek Sparse Attention）做稀疏选择。KV Cache 包含 kv_cache(FP8)、pe_cache(BF16)、indexer_k_cache(FP8) 等组件。

### 对比 V4 的核心差异
- **残差连接**：Residual → mHC（流形约束超连接）
- **注意力层**：MLA/DSA → CSA + HCA 混合注意力
- **优化器**：AdamW → Muon
- **亲和度函数**：Sigmoid → Sqrt(Softplus)
- **后训练**：Mixed RL → Specialist + OPD

### 1M context 全模型 FLOPs（V4 对比基线）
- V3.2：1083.3G FLOPs/token
- V4-Pro：298.6G FLOPs/token（约 3.6x 降幅）

## 关联连接
- [[DeepSeek-V4]] — 下一代模型
- [[DeepSeek-V4-Pro]] — V4 高配版
- [[DeepSeek-V4-Flash]] — V4 轻量版
- [[DeepSeekMoE]] — MoE 架构
- [[Multi-Head Latent Attention]] — MLA 注意力
- [[Multi-Token Prediction]] — MTP 模块
