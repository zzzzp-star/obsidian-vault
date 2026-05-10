---
title: "Generative Reward Model"
type: concept
tags: [奖励模型, RL, 难验证任务, DeepSeek]
sources: [raw/01-articles/DeepSeek-V4技术报告解读-从架构到 Infra 的全栈重构.md]
last_updated: 2026-05-10
---

## 定义
生成式奖励模型（Generative Reward Model, GRM）是 DeepSeek-V4 提出的新型奖励模型范式，完全舍弃传统 scalar reward model（需要大量人工标注），改用 rubric-guided RL data + actor 网络本身兼任 GRM。使得模型的"评判能力"和"生成能力"在同一参数空间中共同进化。

## 关键信息

### 适用场景
"难验证"任务：开放式写作、办公任务、复杂 Agent 行为等——没有 test case 或 rule-based verifier 可以自动评判的任务。

### 核心机制
- **Rubric-Guided**：数据带评分 rubric，模型只需按 rubric 评分而非凭空判断好坏
- **Actor-as-GRM**：RL 训练的 actor 网络同时充当 GRM
- **共同进化**：评判时可调用自身推理能力，生成时又能内化评判标准
- 只需少量多样化人工标注，模型靠自身逻辑泛化到复杂任务

### 与传统 RLHF 的对比
| 维度 | 传统 RLHF | GRM (Actor-as-GRM) |
|------|-----------|---------------------|
| 奖励来源 | 独立训练 scalar RM | Actor 网络自身 |
| 人工标注 | 大量 pairwise 对比 | 少量 rubric 标注 |
| 泛化 | 受限于 RM 训练数据 | 受益于 LLM 推理泛化 |
| 参数效率 | 额外 RM 模型 | 复用已有参数 |

### 潜在风险（自举问题）
弱模型自评可能产生弱 reward，引导 policy 向错误方向。保险措施：
- Rubric 作为评分锚点（不凭空判断）
- 部分易验证任务的 ground truth 约束 GRM 不跑偏

## 关联连接
- [[DeepSeek-V4]] — 应用模型
- [[On-Policy Distillation]] — OPD 蒸馏范式
- [[Specialist训练范式]] — 专家训练
- [[GRPO]] — 策略优化算法
