---
title: "synthesis-deepseek-v4-latency-bandwidth-model"
type: synthesis
tags: [DeepSeek-V4, 推理, 时延建模, KV-Cache, H2D带宽, 性能分析]
sources: [wiki/entities/DeepSeek-V4.md, wiki/entities/DeepSeek-V4-Pro.md, wiki/concepts/Compressed Sparse Attention.md, wiki/concepts/Heavily Compressed Attention.md, wiki/concepts/Lightning Indexer.md, wiki/concepts/KV Cache 管理.md]
last_updated: 2026-05-10
---

# DeepSeek-V4 基于 2 PFLOPS GPU 的时延与带宽建模分析

## 核心问题
对 DeepSeek-V4-Pro 架构在 2 PFLOPS（H100 SXM 级别）GPU 上进行详细的推理时延与 KV Cache 带宽建模，涵盖 6 种序列长度，并分析解码场景下的 H2D 预取带宽需求与 Indexer-Attention 流水线分离的可行性。

---

## 一、架构关键参数

基于 [[DeepSeek-V4-Pro]] 与 [[Heavily Compressed Attention]] 的规格：

| 参数 | 值 |
|------|-----|
| `hidden_dim (d)` | 7168 |
| 层数 | 61 (30 CSA + 31 HCA) |
| 注意力头数 | 128, head_dim=56 |
| CSA 压缩率 / top-k | ratio=4, k=1024 |
| HCA 压缩率 | ratio=128 |
| SWA 窗口 | 128 tokens |
| 激活参数 / 格式 | 49B / FP8 |
| KV Cache @ 1M | ~3.9 GB (V3.2 的 10%) |
| Indexer 精度 | FP4 |
| GPU FP8 算力 | 2 PFLOPS |
| HBM 带宽 / 容量 | 3.35 TB/s / 80 GB |
| PCIe 5.0 ×16 | ~64 GB/s |
| NVLink-C2C | ~900 GB/s |

---

## 二、KV Cache 大小随序列长度变化

依据 V4-Pro 在 1M 时 KV Cache ≈ 3.9 GB，按线性缩放：

| 序列长度 | CSA 压缩块/层 (S/4) | HCA 压缩块/层 (S/128) | 总 KV Cache | HBM 占用率 (80GB) | 单卡最多并发请求数 |
|---------|---------------------|----------------------|-------------|-------------------|-------------------|
| **8k** | 2,000 | 63 | **~31 MB** | 0.04% | ~2,560 |
| **32k** | 8,000 | 250 | **~125 MB** | 0.16% | ~640 |
| **128k** | 32,000 | 1,000 | **~499 MB** | 0.62% | ~160 |
| **300k** | 75,000 | 2,344 | **~1.17 GB** | 1.46% | ~68 |
| **512k** | 128,000 | 4,000 | **~2.00 GB** | 2.50% | ~40 |
| **1M** | 250,000 | 7,813 | **~3.90 GB** | 4.88% | ~20 |

### KV Cache 构成（1M 为例）

| 组件 | 层数 | 精度 | 每层大小 | 合计 |
|------|------|------|---------|------|
| CSA 压缩 KV (K+V latent) | 30 | FP8 | ~96 MB | ~2.88 GB |
| CSA Indexer K | 30 | FP4 | ~16 MB | ~0.48 GB |
| HCA 压缩 KV (K+V latent) | 31 | FP8 | ~6 MB | ~0.19 GB |
| State Cache (SWA + tail) | 61 | BF16 | — | ~0.01 GB |
| **总计** | | | | **~3.9 GB** |

> FP4 的 Indexer K 使 CSA 的索引扫描数据量仅为等量 FP8 的 1/4。

---

## 三、Decoding 单步时延详细建模

### 3.1 计算量 (FLOPs)

| 步骤 | 公式 | 8k | 32k | 128k | 300k | 512k | 1M |
|------|------|-----|------|------|------|------|-----|
| **QKV 投影** (61层) | 61 × 3d² | 9.4G | 9.4G | 9.4G | 9.4G | 9.4G | 9.4G |
| **CSA Indexer QK** (30层, FP4) | 30 × S/4 × d_idx | 3.8M | 15.4M | 61.4M | 144M | 246M | 480M |
| **CSA Top-k Attn** (30层) | 30 × k × d_head × n_heads | 220M | 220M | 220M | 220M | 220M | 220M |
| **HCA Dense Attn** (31层) | 31 × S/128 × d × n_heads | 14M | 56M | 223M | 522M | 892M | **1,736M** |
| **SWA Attn** (61层) | 61 × 128 × d × n_heads | 56M | 56M | 56M | 56M | 56M | 56M |
| **MoE FFN** (61层) | 61 × 6 × 2dd_ff | 16.1G | 16.1G | 16.1G | 16.1G | 16.1G | 16.1G |
| **Output Proj** (61层) | 61 × d² | 3.1G | 3.1G | 3.1G | 3.1G | 3.1G | 3.1G |
| **mHC 连接** | ~5% overhead | 1.4G | 1.4G | 1.4G | 1.4G | 1.4G | 1.4G |
| **总计算量** | | **~30.3G** | **~30.3G** | **~30.6G** | **~30.9G** | **~31.3G** | **~32.1G** |
| **纯计算时延 @2P** | FLOPs/2P | **15.2μs** | **15.2μs** | **15.3μs** | **15.5μs** | **15.7μs** | **16.1μs** |

> 计算时延在 μs 级别，**不是瓶颈**。HCA Dense Attention 是唯一随 S 增长的计算项，但在 1M 时也仅 ~1.7 GFLOPs。

### 3.2 HBM 内存读取量

| 步骤 | 8k | 32k | 128k | 300k | 512k | 1M |
|------|-----|------|------|------|------|-----|
| **模型权重** (49B, FP8) | 49 GB | 49 GB | 49 GB | 49 GB | 49 GB | 49 GB |
| **CSA Indexer 扫描** (30层, FP4) | 7.7 MB | 30.7 MB | 123 MB | 288 MB | 492 MB | 960 MB |
| **CSA Top-k 加载** (30层, FP8) | 24.6 MB | 24.6 MB | 24.6 MB | 24.6 MB | 24.6 MB | 24.6 MB |
| **HCA Dense 加载** (31层, FP8) | 1.5 MB | 5.8 MB | 23.3 MB | 54.6 MB | 93.2 MB | 182 MB |
| **SWA 加载** (61层) | 6 MB | 6 MB | 6 MB | 6 MB | 6 MB | 6 MB |
| **KV 总读取** | **~40 MB** | **~67 MB** | **~177 MB** | **~373 MB** | **~616 MB** | **~1,173 MB** |
| **总 HBM 读取** | **~49.04 GB** | **~49.07 GB** | **~49.18 GB** | **~49.37 GB** | **~49.62 GB** | **~50.17 GB** |
| **内存时延 @3.35TB/s** | **14.64 ms** | **14.65 ms** | **14.68 ms** | **14.74 ms** | **14.81 ms** | **14.98 ms** |

### 3.3 单 Token Decode 总时延（batch=1）

| 序列长度 | 计算时延 | 内存时延 | **实际瓶颈时延** | 瓶颈来源 |
|---------|---------|---------|-----------------|---------|
| **8k** | 15.2 μs | 14.64 ms | **~14.7 ms** | 权重加载 |
| **32k** | 15.2 μs | 14.65 ms | **~14.7 ms** | 权重加载 |
| **128k** | 15.3 μs | 14.68 ms | **~14.7 ms** | 权重加载 |
| **300k** | 15.5 μs | 14.74 ms | **~14.8 ms** | 权重加载 |
| **512k** | 15.7 μs | 14.81 ms | **~14.8 ms** | 权重加载 |
| **1M** | 16.1 μs | 14.98 ms | **~15.0 ms** | 权重加载 |

> **核心结论**：Decoding 完全受限于**模型权重加载**（49 GB FP8），KV Cache 读取即使到 1M 也仅贡献 ~1.2 GB（占总读取的 2.3%）。这正是 DeepSeek-V4 压缩 KV Cache 的核心价值——让 KV 不再成为内存瓶颈。

---

## 四、H2D 预取带宽需求分析

### 4.1 场景定义

- **连续批处理**：batch_i 计算时，同时从 DRAM 向 HBM 预取 batch_{i+1} 的 KV Cache
- **预取窗口** = batch_i 的 decode 步时延
- Batch 大小 B = 32

### 4.2 公式

$$H2D\_BW_{required} = \frac{(1 - hit\_rate) \times KV\_Cache_{batch}}{T_{step}}$$

其中：
- $KV\_Cache_{batch} = B \times KV\_Cache_{per\_request}(S) = 32 \times (S/1M) \times 3.9 \text{ GB}$
- $T_{step} = 14.6 \text{ ms}$（batch=32 时始终受限于权重加载）

### 4.3 计算结果（全量 KV 预取场景）

| 序列长度 | Batch KV 总量 | 70% 命中 (30% miss) | 90% 命中 (10% miss) | 95% 命中 (5% miss) | 99% 命中 (1% miss) |
|---------|--------------|---------------------|---------------------|--------------------|--------------------|
| **8k** | 1.0 GB | 0.30 GB → **20.5 GB/s** ✅ | 0.10 GB → **6.8 GB/s** ✅ | 0.05 GB → **3.4 GB/s** ✅ | 0.01 GB → **0.7 GB/s** ✅ |
| **32k** | 4.0 GB | 1.2 GB → **82 GB/s** ⚠️ | 0.40 GB → **27.4 GB/s** ✅ | 0.20 GB → **13.7 GB/s** ✅ | 0.04 GB → **2.7 GB/s** ✅ |
| **128k** | 16.0 GB | 4.8 GB → **329 GB/s** ⚠️ | 1.6 GB → **110 GB/s** ⚠️ | 0.80 GB → **54.8 GB/s** ✅ | 0.16 GB → **11.0 GB/s** ✅ |
| **300k** | 37.4 GB | 11.2 GB → **769 GB/s** ❌ | 3.7 GB → **256 GB/s** ⚠️ | 1.87 GB → **128 GB/s** ⚠️ | 0.37 GB → **25.6 GB/s** ✅ |
| **512k** | 64.0 GB | 19.2 GB → **1,315 GB/s** ❌ | 6.4 GB → **438 GB/s** ❌ | 3.2 GB → **219 GB/s** ⚠️ | 0.64 GB → **43.8 GB/s** ✅ |
| **1M** | 124.8 GB | 37.4 GB → **2,565 GB/s** ❌ | 12.5 GB → **855 GB/s** ❌ | 6.2 GB → **428 GB/s** ❌ | 1.25 GB → **85.5 GB/s** ⚠️ |

**图例**：✅ PCIe 5.0 ×16 (~64 GB/s) 可达 | ⚠️ 需 NVLink-C2C (~900 GB/s) | ❌ 需多链 NVLink 或 NVSwitch

### 4.4 关键发现

1. **99% HBM 命中率是长上下文的硬门槛**：当 HBM 命中率达到 99%，1M 上下文的 H2D 需求降至 85.5 GB/s，NVLink-C2C 刚好可达。
2. **短上下文（≤32k）宽容度高**：即使 70% 命中率，8k-32k 的需求仍在 PCIe 5.0 范围内。
3. **HBM 容量约束**：80 GB HBM 最多容纳 ~20 个 1M 上下文的完整 KV Cache。Batch=32 必然有 12 个请求需要从 DRAM 调取。
4. **实际建议**：对于 128k+ 上下文，建议设计智能 KV Cache 淘汰策略，维持 ≥95% 的 HBM 命中率；按请求优先级分层（热请求 HBM，冷请求 DRAM）。

---

## 五、Indexer 与 Attention 是否可以分开计算？

### 5.1 结论：可以，这是 DeepSeek-V4 架构的精妙之处

在 [[Compressed Sparse Attention]] 和 [[Lightning Indexer]] 的设计中，CSA 的注意力计算天然分为两个解耦阶段：

```
┌─────────────────────────────────────────────────────────────────┐
│                    CSA Decoding Pipeline                        │
├──────────────┬──────────────────────┬───────────────────────────┤
│   Stage 1    │      Stage 2         │       Stage 3             │
│  INDEXER     │    KV PREFETCH       │     ATTENTION             │
├──────────────┼──────────────────────┼───────────────────────────┤
│ Q → c_Q 投影  │ 根据 Top-k 索引       │ 加载已缓存的 Top-k KV块   │
│ Indexer QK   │ 发起 DRAM→HBM 预取   │ Q·K^T + Softmax + ·V     │
│ (FP4, 全扫描) │ (仅取 Top-k 块的KV)  │ + SWA 局部注意力          │
│ Top-k 选择   │                      │ + 输出投影                │
├──────────────┼──────────────────────┼───────────────────────────┤
│ 仅需 Indexer │ 仅需 Top-k 索引列表   │ 需要完整 KV (FP8)         │
│ KV (FP4,小)  │ + H2D 带宽           │ 已驻留在 HBM 中            │
└──────────────┴──────────────────────┴───────────────────────────┘
```

### 5.2 为什么可以分离？

**数据依赖分析**：

1. **Indexer 仅依赖**：当前 token 的 query（已知）+ Indexer-KV cache（FP4，极小，通常全部驻留 HBM）。**不依赖** attention 的输出或完整的压缩 KV。
2. **Attention 仅依赖**：Top-k 索引（Indexer 的输出）+ 对应的 Top-k 完整 KV 块（需在 HBM 中）。
3. **无循环依赖**：Indexer 不消费 attention 输出，attention 不修改 Indexer 状态。

### 5.3 流水线化的预取策略

```
时间轴 →

Batch_i:     |── Indexer(i) ──|── KV Prefetch(i) ──|──── Attention(i) + FFN(i) ────|
Batch_{i+1}:                   |── Indexer(i+1) ──|── KV Prefetch(i+1) ──|── Attn(i+1) ...

重叠区域:     [═ Batch_i 的 Attention/FFN ════════╗
              ║ Batch_{i+1} 的 Indexer + Prefetch ═╝
```

| 时间 | Batch_i 做什么 | Batch_{i+1} 并行做什么 |
|------|---------------|----------------------|
| T1 | Attention(CSA top-k) + FFN | **Indexer**：QK score (FP4) 全扫描 S/4 个压缩块 |
| T2 | Attention(HCA dense) + FFN | **Top-k 选择**：选出得分最高的 1024 个块 |
| T3 | 输出投影 | **发起 H2D 预取**：仅预取这 1024 个块的完整 KV |
| T4 | (完成) | KV 块已就绪，开始 Attention |

### 5.4 Top-k 预取量优化

关键洞察：**不需要预取全部 KV Cache，只需预取 Indexer 选出的 top-k 块！**

| 序列长度 | 全部 KV Cache (per request) | Top-k 块 KV (k=1024) | 预取缩减比 |
|---------|---------------------------|---------------------|-----------|
| **8k** | 31 MB | 0.8 MB | **38:1** |
| **32k** | 125 MB | 0.8 MB | **156:1** |
| **128k** | 499 MB | 0.8 MB | **624:1** |
| **300k** | 1.17 GB | 0.8 MB | **1,462:1** |
| **512k** | 2.00 GB | 0.8 MB | **2,500:1** |
| **1M** | 3.90 GB | 0.8 MB | **4,875:1** |

> Top-k=1024 块的 KV 大小固定为 ~0.8 MB（30 CSA 层 × 1024 × d_kv_latent × 2），与序列长度无关！

### 5.5 修正后的 H2D 带宽需求

采用 Indexer 先行的 top-k 预取策略后，CSA 部分仅需预取 0.8 MB，HCA 部分全量预取（1M 时 182 MB/请求）：

| 序列长度 | 需预取总量 (HCA全量 + CSA top-k) | 95% 命中所需带宽 | 90% 命中所需带宽 |
|---------|-------------------------------|-----------------|-----------------|
| **1M** | 182 MB + 0.8 MB ≈ **183 MB/req** | 5% × 32 × 183 MB / 14.6 ms = **20.1 GB/s** ✅ | 10% × 32 × 183 MB / 14.6 ms = **40.1 GB/s** ✅ |

即使 90% 命中率，PCIe 5.0 ×16（64 GB/s）也完全够用！

### 5.6 为什么有效——架构视角

这依赖于 DeepSeek-V4 的三层注意力分工：

- **HCA**（[[Heavily Compressed Attention]]）："商范畴的 Nerve 构造"——全局低分辨率概览，压缩率 128:1，全量 KV 仅 182 MB（1M 上下文），全量预取也不是负担
- **CSA**（[[Compressed Sparse Attention]]）："保持同伦等价的稀疏化"——中等分辨率 + top-k 选择，通过 Indexer 先确定"骨架边"再加载，只需预取 0.8 MB
- **SWA**：局部 128 token 的细节，KV 可忽略不计

三者分而治之，Indexer 天然充当"预取预测器"，使得流水线化预取不仅可行，而且带宽需求极为温和。

---

## 六、总结

| 维度 | 结论 |
|------|------|
| **计算瓶颈** | Decoding 完全受限于权重加载（~14.6 ms/step），非计算瓶颈 |
| **KV Cache 效率** | V4-Pro 1M 上下文仅 3.9 GB，KV 读取仅占总内存读取 2.3% |
| **H2D 预取** | 维持 ≥95% HBM 命中率是关键；128k+ 需 NVLink 级别带宽 |
| **Indexer-Attention 分离** | ✅ 可行，Indexer 天然充当 KV 预取预测器 |
| **Top-k 预取优化** | 将预取量从 3.9 GB 降至 0.8 MB（CSA 部分），H2D 带宽需求降至 PCIe 5.0 范围 |
| **架构优势** | CSA(稀疏选择) + HCA(全量低分辨率) + SWA(局部细节) 的三层设计，使 Indexer-Attention 分离成为一个自然且高效的流水线优化策略 |

---

## 关联连接
- [[DeepSeek-V4]] — 主体模型架构
- [[DeepSeek-V4-Pro]] — 建模目标规格
- [[Compressed Sparse Attention]] — CSA 稀疏注意力机制
- [[Heavily Compressed Attention]] — HCA 重度压缩注意力
- [[Lightning Indexer]] — Indexer 设计与 Indexer-Attention 分离的可行性基础
- [[KV Cache 管理]] — KV Cache 混合精度存储与分级设计
