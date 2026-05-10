---
title: "摘要-deepseek-v4-full-stack-review"
type: source
tags: [DeepSeek, 模型架构, 基础设施, 后训练, Agent]
sources: [raw/01-articles/DeepSeek-V4技术报告解读-从架构到 Infra 的全栈重构.md]
last_updated: 2026-05-10
---

## 核心摘要
本文是对 DeepSeek-V4 技术报告的全面解读，覆盖从架构到 Infra 的全栈重构。核心叙事是将 1M token 上下文变为日常可用。架构层面：mHC 约束残差连接、CSA+HCA 混合注意力、Muon 优化器替代 AdamW。Infra 层面：MegaMoE 超融合 EP kernel、TileLang kernel DSL、FP4 量化感知训练、分层 KV Cache 与磁盘前缀复用。后训练采用 Specialist + On-Policy Distillation (OPD) 两阶段范式替代混合 RL，并引入 Generative Reward Model (GRM) 处理难验证任务。Agent 能力的核心竞争力从"数据菜谱"转向"基础设施"——DSec 沙箱平台提供 execution authenticity、trajectory reproducibility、reward accessibility、capability mergeability 四大支柱。

## 关联连接
- [[DeepSeek-V4]] — 主体模型
- [[DeepSeek-V4-Pro]] — Pro 版本
- [[DeepSeek-V4-Flash]] — Flash 版本
- [[MegaMoE]] — 超融合 EP kernel
- [[On-Policy Distillation]] — 在线策略蒸馏
- [[Generative Reward Model]] — 生成式奖励模型
- [[DSec]] — 沙箱平台
- [[Specialist训练范式]] — 专家训练范式
- [[FP4 量化感知训练]] — FP4 QAT
- [[TileLang]] — Kernel DSL
