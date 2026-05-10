---
title: "DeepSeek-V4"
type: entity
tags: [DeepSeek, LLM, MoE, 长上下文, 开源模型]
sources: [raw/01-articles/zartbot-DeepSeek-V4详细分析（1）：算法和模型结构.md, raw/01-articles/DeepSeek-V4技术报告解读-从架构到 Infra 的全栈重构.md, raw/02-papers/DeepSeek_V4.pdf]
last_updated: 2026-05-10
---

## 定义
DeepSeek-V4 是 DeepSeek-AI 于 2025 年发布的大语言模型系列，核心目标是实现百万级 Token 上下文的高效智能处理。技术报告标题为《DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence》。该系列包含两个 MoE 模型：DeepSeek-V4-Pro 和 DeepSeek-V4-Flash。

## 关键信息

### 模型规格
| 指标 | DeepSeek-V4-Flash | DeepSeek-V4-Pro |
|------|-------------------|-----------------|
| 总参数 | 284B | 1.6T |
| 激活参数 | 13B | 49B |
| hidden_dim | 4096 | 7168 |
| 层数 | 43 | 61 |
| 路由专家数 | 256 | 384 |
| 激活专家数 | 6 | 6 |
| 专家中间层维度 | 2048 | 3072 |

### 三大架构创新
1. **mHC (Manifold-Constrained Hyper-Connections)**：替代传统残差连接，将变换矩阵约束到双随机矩阵流形上，增强深层训练的数值稳定性
2. **混合注意力 (CSA + HCA)**：CSA（压缩稀疏注意力，ratio=4）+ HCA（重度压缩注意力，ratio=128）交替排列，配合滑动窗口注意力，在保持长距离依赖的同时大幅降低计算量和 KV Cache
3. **Muon 优化器**：替代 AdamW 作为主优化器，通过 Newton-Schulz 迭代将更新矩阵投影到 Stiefel 流形

### 效率突破（vs DeepSeek-V3.2，1M context）
- V4-Pro：单 token FLOPs 降至 27%，KV Cache 降至 10%
- V4-Flash：单 token FLOPs 降至 10%，KV Cache 降至 7%

### 后训练范式
采用 **Specialist + On-Policy Distillation (OPD)** 两阶段取代混合 RL：
1. Specialist Training：针对 math/code/agent/IF 各领域独立训练专家
2. OPD：学生模型通过 reverse KL 从多个 teacher 的 full-vocabulary logits 蒸馏

### 三档推理强度
- **Non-think**：快速直觉响应
- **Think High**：自觉逻辑分析
- **Think Max**：极限推理展开

## 关联连接
- [[DeepSeek-V4-Pro]] — Pro 版本详情
- [[DeepSeek-V4-Flash]] — Flash 版本详情
- [[DeepSeek-V3.2]] — 前代模型
- [[Compressed Sparse Attention]] — CSA 注意力机制
- [[Heavily Compressed Attention]] — HCA 注意力机制
- [[Manifold-Constrained Hyper-Connections]] — mHC 残差连接
- [[Muon Optimizer]] — 优化器
- [[DeepSeekMoE]] — MoE 架构
- [[On-Policy Distillation]] — OPD 蒸馏范式
- [[Generative Reward Model]] — GRM 奖励模型
- [[MegaMoE]] — Infra 加速
- [[FP4 量化感知训练]] — 量化技术
- [[DSec]] — 沙箱平台
