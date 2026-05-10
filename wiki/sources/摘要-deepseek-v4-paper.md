---
title: "摘要-deepseek-v4-paper"
type: source
tags: [DeepSeek, 论文, 模型架构, 长上下文]
sources: [raw/02-papers/DeepSeek_V4.pdf]
last_updated: 2026-05-10
---

## 核心摘要
DeepSeek-V4 官方技术报告《DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence》。该报告正式发布了 DeepSeek-V4 系列的两个 MoE 模型：Pro（1.6T/49B）和 Flash（284B/13B），原生支持 1M token 上下文。核心贡献：(1) mHC 增强深层残差连接；(2) CSA+HCA 混合注意力实现长上下文效率突破；(3) Muon 优化器；(4) Specialist + OPD 后训练范式；(5) MegaMoE/TileLang/FP4 QAT 等 Infra 创新。在 1M context 下 V4-Pro 仅需 V3.2 的 27% FLOPs 和 10% KV Cache。

## 关联连接
- [[DeepSeek-V4]] — 模型总览
- [[Compressed Sparse Attention]] — CSA
- [[Heavily Compressed Attention]] — HCA
- [[Manifold-Constrained Hyper-Connections]] — mHC
- [[DeepSeekMoE]] — MoE 架构
- [[DeepSeek-V3.2]] — 前代模型
