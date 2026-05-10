---
title: "摘要-llm-inference-parallel-strategies"
type: source
tags: [推理, 并行策略, 分布式, LLM]
sources: [raw/01-articles/大模型推理并行策略(DP,TP,PP,SP,EP)原理简介.md]
last_updated: 2026-05-10
---

## 核心摘要
本文系统介绍了大模型推理部署中的主要并行策略。根据输入激活值的切分维度分类：切 batch 为 DP（数据并行）、切序列为 SP/CP（序列并行/上下文并行）、切隐藏层尺寸为 TP（张量并行）、按层切分为 PP（流水线并行）、按专家切分为 EP（专家并行）。每种策略各有适用场景：DP 处理多请求并发、TP 解决单层参数过大、SP 处理长序列、PP 拆分极深模型、EP 为 MoE 模型专用。实际部署需结合模型参数量、PD/AF 分离需求、硬件拓扑等因素综合选用。

## 关联连接
- [[大模型推理并行策略]] — 综合概念页
- [[Data Parallelism]] — 数据并行
- [[Tensor Parallelism]] — 张量并行
- [[Pipeline Parallelism]] — 流水线并行
- [[Sequence Parallelism]] — 序列并行
- [[Expert Parallelism]] — 专家并行
- [[Context Parallelism]] — 上下文并行
- [[DeepSeek-V4]] — DeepSeek-V4 推理实践
