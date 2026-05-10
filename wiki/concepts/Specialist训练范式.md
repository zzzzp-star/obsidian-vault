---
title: "Specialist训练范式"
type: concept
tags: [后训练, RL, 专家模型, DeepSeek]
sources: [raw/01-articles/DeepSeek-V4技术报告解读-从架构到 Infra 的全栈重构.md]
last_updated: 2026-05-10
---

## 定义
Specialist 训练范式是 DeepSeek-V4 后训练两阶段流程的第一阶段。针对 math、code、agent、instruction following 等每个目标领域，独立训练一个专家模型（SFT + GRPO RL），然后通过 OPD 蒸馏整合为统一模型。

## 关键信息

### 两阶段流程
1. **Specialist Training**：每个领域独立训练专家（SFT → GRPO RL）
2. **On-Policy Distillation**：学生模型从多个 teacher 的 full-vocabulary logits 蒸馏

### 三档推理强度（Reasoning Effort）
| 模式 | 特点 | 训练差异 |
|------|------|----------|
| Non-think | 快速直觉响应 | 短 length penalty |
| Think High | 自觉逻辑分析 | 中等 length penalty |
| Think Max | 极限推理展开 | 长 context window + 特殊 system prompt |

### 核心优势
- 每个专家可用最适合自己的 reward（数学用 rule-based，code 用 test case，写作/Agent 用 rubric-GRM）
- 避免混合 RL 中不同域 reward hack 互相干扰
- OPD 让学生"选择性"靠拢相关 teacher，而非无差别平均

### Agent 训练基础设施
- **DSec 沙箱**：支持 Function Call/Container/microVM/fullVM 四级执行环境
- **Trajectory Logging**：全序轨迹日志，支持 fast-forwarding replay
- **Preemptible Rollout**：Token 级 WAL + KV cache 保存，抢占恢复
- **Rule-based Verifier + GRM**：易验证用规则，难验证用 actor-as-GRM

## 关联连接
- [[On-Policy Distillation]] — OPD 蒸馏阶段
- [[Generative Reward Model]] — GRM 奖励模型
- [[DSec]] — 沙箱平台
- [[DeepSeek-V4]] — 应用模型
- [[GRPO]] — 策略优化算法
