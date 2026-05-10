---
title: "MegaMoE"
type: concept
tags: [Infra, MoE, EP, 通信优化, DeepSeek]
sources: [raw/01-articles/DeepSeek-V4技术报告解读-从架构到 Infra 的全栈重构.md]
last_updated: 2026-05-10
---

## 定义
MegaMoE 是 DeepSeek-V4 设计的超融合 EP（Expert Parallelism）kernel，将 MoE 的 All-to-All 通信与计算融合为一个 kernel。通过将 expert 拆成 wave，实现"计算当前 wave + 发送上一 wave + 接收下一 wave"的三股并行流水线，在 V4-Flash 架构下理论加速 1.92×。

## 关键信息

### 三种 EP 方案对比
| 方案 | 特点 | 效率 |
|------|------|------|
| Naive Solution | Dispatch→L1→Act→L2→Combine 完全串行 | 基准 |
| Comet | Dispatch↔L1、L2↔Combine 两段两两重叠 | 中等 |
| MegaMoE (V4) | Wave 1/2/3 流水，四通道完全 overlap | 1.92× 理论加速 |

### 实测性能
- NVIDIA GPU：1.5–1.73× speedup
- 华为 Ascend NPU：类似加速比
- RL rollout 长尾小 batch 场景：可达 1.96×

### 硬件洞察
- Compute/Bandwidth 比值是决定完全 overlap 的关键指标
- V4-Pro 每 6.1 TFLOP/s 对应 1 GBps 带宽即可，再加带宽边际收益递减
- 建议未来硬件给功耗留足余量（融合 kernel 同时拉满算+存+网）
- 建议将 SwiGLU 换成无 exp/无 division 的激活函数

### 开源
已作为 DeepGEMM 的一部分开源。

## 关联连接
- [[DeepSeek-V4]] — 主体模型
- [[DeepSeekMoE]] — MoE 架构
- [[Expert Parallelism]] — 专家并行策略
- [[TileLang]] — kernel DSL
