---
title: "DeepSeekMoE"
type: concept
tags: [MoE, 路由, 负载均衡, DeepSeek, 模型架构]
sources: [raw/01-articles/zartbot-DeepSeek-V4详细分析（1）：算法和模型结构.md, raw/01-articles/DeepSeek-V4技术报告解读-从架构到 Infra 的全栈重构.md]
last_updated: 2026-05-10
---

## 定义
DeepSeekMoE 是 DeepSeek 系列模型的混合专家（Mixture-of-Experts）架构。采用细粒度路由专家 + 共享专家的设计，通过无辅助损失（auxiliary-loss-free）负载均衡策略和精心设计的路由机制，实现参数规模与计算效率的平衡。

## 关键信息

### 架构组成
- **路由专家（Routed Experts）**：细粒度专家，每个 token 根据路由选择激活部分专家
- **共享专家（Shared Expert）**：所有 token 共享，捕获通用知识
- V4-Pro：384 路由专家 + 1 共享专家，激活 6 个
- V4-Flash：256 路由专家 + 1 共享专家，激活 6 个

### 路由亲和度函数演进
| 版本 | 函数 | 特点 |
|------|------|------|
| DeepSeek-V3 | Sigmoid | 非竞争性，各自独立评分，比 Softmax 减少专家坍缩 |
| DeepSeek-V4 | Sqrt(Softplus) | 无上界，区分度更好，梯度永不饱和，Sqrt 防止分数差异过大 |

### Sqrt(Softplus) 设计分析
- Softplus 输出范围 [0, +∞)，无上界，高匹配专家获得更高区分度
- Sqrt 压缩防止分数差异过大导致路由权重集中
- 导数永不归零，即使路由器已对某专家产生极强亲和度仍能接收有效梯度
- TileLang kernel 中设阈值 20 避免 softplus 过大

### V4 微调整
- 移除对路由目标节点数量的限制，重新设计并行策略
- 前几层采用哈希路由（Hash Routing）：根据 token ID 的哈希函数决定专家，不走 token-wise gating
- 负载均衡：无辅助损失 + 轻微序列级平衡损失防止单序列内极端不平衡

### 哈希路由
```python
self.tid2eid = nn.Parameter(
    torch.empty(vocab_size, n_activated_experts, dtype=torch.int32),
    requires_grad=False,
)
```
固定的 hash 表，通过 token ID 查询需激活的专家，无需学习路由参数。

## 关联连接
- [[DeepSeek-V4]] — 最新应用
- [[DeepSeek-V3.2]] — V3.2 的 MoE 设计
- [[MegaMoE]] — EP 加速 kernel
- [[Expert Parallelism]] — 专家并行策略
