---
title: "DeepSeek-V4-Pro"
type: entity
tags: [DeepSeek, LLM, MoE, 推理模型]
sources: [raw/01-articles/DeepSeek-V4技术报告解读-从架构到 Infra 的全栈重构.md, raw/01-articles/zartbot-DeepSeek-V4详细分析（1）：算法和模型结构.md]
last_updated: 2026-05-10
---

## 定义
DeepSeek-V4-Pro 是 DeepSeek-V4 系列的高配版本，总参数 1.6T，激活参数 49B，拥有 61 层 Transformer、384 个路由专家（激活 6 个）、hidden_dim=7168。在编程和 Agent 任务上基本持平 Claude Opus 4.6，API 价格远低于闭源竞品。

## 关键信息

### 核心参数
- 层数：61 层（31 HCA + 30 CSA + 1 MTP）
- hidden_dim：7168
- 注意力头数：128
- CSA top-k：1024
- mHC n_hc：4

### 架构配置
- CSA 层（ratio=4，30 层）：中等压缩 + 稀疏选择，启用重叠窗口
- HCA 层（ratio=128，31 层）：重度压缩 + 密集注意力
- SWA 窗口大小：128 token

### 推理效率（1M context vs V3.2）
- 单 token FLOPs：仅为 V3.2 的 27%
- KV Cache 大小：仅为 V3.2 的 10%
- 比 BF16 GQA8 基线节省约 98% KV Cache

### Benchmark 表现
- Codeforces Rating 3206（与 GPT-5.4-xHigh 持平，人类选手第 23 名）
- SWE-Verified 80.6
- SimpleQA-Verified 57.9（开源第一）
- LiveCodeBench 93.5

### API 定价（每百万 Token USD）
- 输入：$1.74
- 缓存命中输入：$0.145
- 输出：$3.48

## 关联连接
- [[DeepSeek-V4]] — 系列总览
- [[DeepSeek-V4-Flash]] — Flash 版本
- [[DeepSeek-V3.2]] — 前代模型
- [[Compressed Sparse Attention]] — CSA 机制
- [[Heavily Compressed Attention]] — HCA 机制
- [[DeepSeekMoE]] — MoE 架构
