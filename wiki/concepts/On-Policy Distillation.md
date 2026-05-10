---
title: "On-Policy Distillation"
type: concept
tags: [后训练, 蒸馏, RL, 策略优化]
sources: [raw/01-articles/DeepSeek-V4技术报告解读-从架构到 Infra 的全栈重构.md]
last_updated: 2026-05-10
---

## 定义
在线策略蒸馏（On-Policy Distillation, OPD）是 DeepSeek-V4 后训练的核心范式，替代了 V3.2 的混合 RL（Mixed RL）。学生模型在自己采样的轨迹上，从多个领域专家 teacher 的 full-vocabulary logits 分布中通过 reverse KL 散度进行蒸馏学习。

## 关键信息

### OPD vs Mixed RL
- **V3.2 Mixed RL**：所有领域（math/code/agent/IF）混在一起做 RL 训练，存在不同域 reward hack 互相干扰的风险
- **V4 OPD**：先独立训练各领域专家（Specialist），再通过 OPD 蒸馏回统一学生模型

### Reverse KL 关键点
- 轨迹从学生自己采样（保持 on-policy），避免 off-policy 的分布偏移
- 学生会"选择性"地靠近相关任务的 teacher：数学题靠向数学 teacher，代码题靠向代码 teacher
- 相比传统 weight merging，在 logits 级对齐更稳，有效避免"能力抵消"

### Full-Vocabulary Logits
- 不退化到 per-token KL 估计（梯度方差高、训练不稳）
- 坚持做 full-vocab KL，需大量工程优化

### 工程实现
- Teacher 权重集中存分布式存储，按需 ZeRO-like 参数分片加载
- 只缓存最后一层 hidden states，训练时通过 prediction head 现场算 logits
- 按 teacher index 排序训练样本，每个 mini-batch 最多一个 teacher head 驻留显存
- Teacher 参数、hidden states 的 load/offload 全异步

## 关联连接
- [[DeepSeek-V4]] — 应用模型
- [[Specialist训练范式]] — Specialist 训练阶段
- [[Generative Reward Model]] — GRM 奖励模型
- [[GRPO]] — 专家 RL 训练算法
