---
title: "摘要-deepseek-v3-2-paper"
type: source
tags: [DeepSeek, 论文, MoE, 稀疏注意力]
sources: [raw/02-papers/DeepSeek_V3_2.pdf]
last_updated: 2026-05-10
---

## 核心摘要
DeepSeek-V3.2 是 DeepSeek-V4 的直接前代模型，采用 DeepSeekMoE 架构 + DSA（DeepSeek Sparse Attention）+ Multi-Token Prediction（MTP）。相比 V3，V3.2 在推理效率和上下文长度上已有显著提升。其 MLA（Multi-Head Latent Attention）通过 KV 低秩压缩实现了 KV Cache 的大幅缩减。DeepSeek-V4 在 V3.2 的基础上进行了三大关键替换：残差连接从 Residual 升级为 mHC，注意力层从 MLA/DSA 升级为 CSA+HCA 混合注意力，优化器从 AdamW 升级为 Muon。

## 关联连接
- [[DeepSeek-V3.2]] — 模型介绍
- [[DeepSeek-V4]] — 下一代模型
- [[DeepSeekMoE]] — MoE 架构
- [[Multi-Head Latent Attention]] — MLA 注意力
- [[Multi-Token Prediction]] — MTP 模块
