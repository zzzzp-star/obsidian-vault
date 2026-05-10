---
title: "DeepSeek-V4-Flash"
type: entity
tags: [DeepSeek, LLM, MoE, 轻量模型]
sources: [raw/01-articles/DeepSeek-V4技术报告解读-从架构到 Infra 的全栈重构.md, raw/01-articles/zartbot-DeepSeek-V4详细分析（1）：算法和模型结构.md]
last_updated: 2026-05-10
---

## 定义
DeepSeek-V4-Flash 是 DeepSeek-V4 系列的轻量版本，总参数 284B，激活参数仅 13B，拥有 43 层 Transformer、256 个路由专家（激活 6 个）、hidden_dim=4096。以极低的推理成本提供接近 Pro 版的推理能力。

## 关键信息

### 核心参数
- 层数：43 层
- hidden_dim：4096
- 注意力头数：64
- CSA top-k：512
- mHC n_hc：4

### 效率极致（1M context vs V3.2）
- 单 token FLOPs：仅为 V3.2 的 10%
- KV Cache 大小：仅为 V3.2 的 7%

### 能力定位
- V4-Flash-Max 在推理类任务上能赶上 Pro-High 甚至部分场景接近 Pro-Max
- 世界知识上与 Pro 差距明显（参数体量限制）
- Agent 高难度任务（如 Terminal Bench）明显不如 Pro

### API 定价（每百万 Token USD）
- 输入：$0.14
- 缓存命中输入：$0.028
- 输出：$0.28

## 关联连接
- [[DeepSeek-V4]] — 系列总览
- [[DeepSeek-V4-Pro]] — Pro 版本
- [[DeepSeek-V3.2]] — 前代模型
- [[Compressed Sparse Attention]] — CSA 机制
- [[Heavily Compressed Attention]] — HCA 机制
