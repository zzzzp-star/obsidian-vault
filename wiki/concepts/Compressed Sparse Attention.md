---
title: "Compressed Sparse Attention"
type: concept
tags: [注意力机制, 稀疏注意力, KV压缩, 长上下文, DeepSeek]
sources: [raw/01-articles/zartbot-DeepSeek-V4详细分析（1）：算法和模型结构.md, raw/01-articles/DeepSeek-V4技术报告解读-从架构到 Infra 的全栈重构.md]
last_updated: 2026-05-10
---

## 定义
压缩稀疏注意力（Compressed Sparse Attention, CSA）是 DeepSeek-V4 提出的高效注意力机制，通过将 KV Cache 先压缩再稀疏选择的两阶段策略，大幅降低长上下文的计算和存储开销。压缩率 m=4（每 4 个 token 压缩为 1 个条目），然后通过 Lightning Indexer 在压缩条目上选择 top-k 个进行核心注意力计算。

## 关键信息

### 核心流程
1. **KV 压缩**：将每 m=4 个连续 token 的 KV 缓存通过加权求和压缩为 1 个摘要向量
2. **稀疏选择**：Lightning Indexer 计算查询与所有压缩块的"相关性得分"，top-k 选择最高的进行注意力计算
3. **核心注意力**：以 MQA 方式在选中的压缩 KV 条目上执行注意力

### 技术细节
- 启用重叠窗口（overlap=True, coff=2），相邻压缩块共享边界 token，减少信息断裂
- 压缩权重通过 softmax 归一化的门控评分确定
- 可学习位置偏置（APE）增强压缩质量
- 压缩后附加 128 token 滑动窗口注意力作为局部细节补丁

### 压缩流水线
```
hidden_states → wkv 投影 → gate 评分 + APE 位置编码 → softmax 加权求和 → RMSNorm → RoPE → 量化 → 写入 KV Cache
```

### Lightning Indexer
- 查询下投影得到压缩潜向量，再上投影出多头 indexer query
- Indexer 的 QK 计算使用 FP4 精度加速
- Indexer 与主注意力共享潜查询向量，避免重复投影

### 范畴论视角
CSA 被类比为"保持同伦等价的稀疏化"：先构造商范畴（中等分辨率抽象），再通过 top-k 选择保留"骨架边"。它提供了一张"中等分辨率的全局交通图"，忽略毛细血管但保留主干道。

## 关联连接
- [[Heavily Compressed Attention]] — 互补的重度压缩注意力
- [[DeepSeek-V4]] — 主体模型
- [[Lightning Indexer]] — 稀疏选择索引器
- [[KV Cache 管理]] — KV 缓存管理
- [[Multi-Head Latent Attention]] — MLA 注意力（V3 前身）
