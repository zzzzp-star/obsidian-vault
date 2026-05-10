---
title: "Heavily Compressed Attention"
type: concept
tags: [注意力机制, KV压缩, 长上下文, DeepSeek, 密集注意力]
sources: [raw/01-articles/zartbot-DeepSeek-V4详细分析（1）：算法和模型结构.md, raw/01-articles/DeepSeek-V4技术报告解读-从架构到 Infra 的全栈重构.md]
last_updated: 2026-05-10
---

## 定义
重度压缩注意力（Heavily Compressed Attention, HCA）是 DeepSeek-V4 提出的高效注意力机制，采用极激进的压缩率 m'=128（每 128 个 token 压缩为 1 个条目），在压缩后的条目上执行密集注意力（Dense Attention），不进行稀疏选择。HCA 提供"低分辨率的全局概览图"，捕捉文档的宏观板块结构。

## 关键信息

### 核心流程
1. **重度压缩**：将每 m'=128 个连续 token 的 KV 缓存压缩为 1 个摘要向量，不启用重叠窗口
2. **密集注意力**：在所有压缩条目之间执行全注意力（dense attention），不做稀疏选择
3. **核心注意力**：以 MQA 方式在压缩 KV + 128 token SWA 上执行

### 技术特点
- 压缩率远大于 CSA（128 vs 4），压缩强度更高
- 无需 Indexer 做稀疏选择，直接 dense attention
- 序列有效长度缩减为 n/128，dense attention 开销可接受
- 同样附加 128 token 滑动窗口注意力分支

### 范畴论视角
HCA 被类比为"商范畴的 Nerve 构造"：
- 将原始 token 按 128 分组，定义等价关系
- 构建商范畴，其对象是"token 块"而非单个 token
- 在商范畴上执行密集注意力，相当于计算块与块之间的宏观关系
- HCA 主要捕捉 0 维同调（全局连通分支），确保相距遥远的宏观概念连接性不被忽略

### 与 CSA 的交替配置
DeepSeek-V4 在不同层交替使用 CSA 和 HCA：
- HCA 层：获得文本"大陆板块"分布的全局认知
- CSA 层：利用全局认知，在中等尺度上有方向性地寻找关键连接
- SWA：补充局部细节

### 量化数据（V4-Pro）
- HCA(ratio=128) 层数：31
- CSA(ratio=4) 层数：30
- 1M context 下 HCA 压缩后的注意力计算仅为 33.54M + 2048×Seq FLOPs/token

## 关联连接
- [[Compressed Sparse Attention]] — 互补的压缩稀疏注意力
- [[DeepSeek-V4]] — 主体模型
- [[KV Cache 管理]] — KV 缓存管理
- [[商范畴]] — 数学理论基础
