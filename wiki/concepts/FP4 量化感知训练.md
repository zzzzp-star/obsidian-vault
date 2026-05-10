---
title: "FP4 量化感知训练"
type: concept
tags: [量化, FP4, 训练, 推理, DeepSeek]
sources: [raw/01-articles/DeepSeek-V4技术报告解读-从架构到 Infra 的全栈重构.md]
last_updated: 2026-05-10
---

## 定义
FP4 量化感知训练（FP4 Quantization-Aware Training, QAT）是 DeepSeek-V4 在预训练后期引入的低精度训练技术。采用 MXFP4 格式覆盖两个关键路径：MoE expert 权重和 CSA indexer QK 路径。核心工程巧思是 FP4→FP8 无损反量化，可复用现有 FP8 训练框架。

## 关键信息

### 覆盖路径
1. **MoE expert 权重**（占 GPU 内存大头）：训练时用 FP4 存储，推理时直接用 FP4
2. **CSA indexer QK 路径**（attention 评分计算热点）：FP4 QAT 加速

### FP4 → FP8 无损反量化
- FP8（E4M3）比 FP4（E2M1）多 2 位 exponent
- 只要 128×128 FP8 块内的 FP4 子块（1×32）scale 比值不超过阈值
- 细粒度 scale 信息可被 FP8 动态范围完全吸收
- 整个 QAT pipeline 直接复用 FP8 训练框架

### 训练流程
- 预训练后期引入 FP4 QAT
- 梯度对 FP8 权重求并直接回传 FP32 master weights（STE 直通估计）
- RL rollout 和推理阶段直接用真正 FP4 权重
- 训练与部署行为一致

### 额外优化
- Index score 从 FP32 量化到 BF16，top-k selector 提速 2×，KV recall 仍保持 99.7%

## 关联连接
- [[DeepSeek-V4]] — 主体模型
- [[Lightning Indexer]] — 受益于 FP4 QAT
- [[DeepSeekMoE]] — MoE 权重 FP4 量化
- [[MegaMoE]] — 量化后加速推理
