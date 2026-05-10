---
title: "synthesis-deepseek-v4-latency-bandwidth-model"
type: synthesis
tags: [DeepSeek-V4, 推理, 时延建模, KV-Cache, H2D带宽, 性能分析]
sources: [wiki/entities/DeepSeek-V4.md, wiki/entities/DeepSeek-V4-Pro.md, wiki/concepts/Compressed Sparse Attention.md, wiki/concepts/Heavily Compressed Attention.md, wiki/concepts/Lightning Indexer.md, wiki/concepts/KV Cache 管理.md, raw/01-articles/zartbot-DeepSeek-V4详细分析：数据表格.md, raw/09-archive/DeepSeek_V4.pdf]
last_updated: 2026-05-11
---

# DeepSeek-V4 基于 2 PFLOPS GPU 的时延与带宽建模分析

## 核心问题
对 DeepSeek-V4-Pro 架构在 2 PFLOPS（H100 SXM 级别）GPU 上进行详细的推理时延与 KV Cache 带宽建模，涵盖 6 种序列长度，并分析解码场景下的 H2D 预取带宽需求与 Indexer-Attention 流水线分离的可行性。本次建模基于 [[摘要-deepseek-v4-zartbot-analysis]] 中的精确 FLOPs 分解公式与 KV Cache 结构参数。

---

## 零、运算量与 KVCache 整体对比（柱状图）

### FLOPs 对比

![[deepseek-v4-flops-comparison.png]]

> 数据来源：[[raw/01-articles/zartbot-DeepSeek-V4详细分析：数据表格.md]] 运算量对比表。V4-Pro 的每 token FLOPs 随序列长度增长极为平缓：从 8K 的 109.3G 仅增至 1M 的 298.6G（仅 2.73×），而 V3.2 从 91.7G 飙升至 1083.3G（11.8×）。在 1M 上下文，V4-Pro 仅为 V3.2 的 27.6%。

### KV Cache 对比

![[deepseek-v4-kvcache-comparison.png]]

> 数据来源：[[raw/01-articles/zartbot-DeepSeek-V4详细分析：数据表格.md]] KVCache 用量对比表。V4-Pro 的 KV Cache 在长上下文下仅为 V3.2 的 ~10%，且节省比随序列长度增加而收敛至 10.5×。1M 时 V4-Pro 仅需 4.46 GB（V3.2 需 47 GB）。

---

## 一、架构关键参数

基于 [[DeepSeek-V4-Pro]]、[[Heavily Compressed Attention]] 及技术报告的精确参数：

| 参数 | 符号 | 值 |
|------|------|-----|
| 隐藏维度 | `d` | 7168 |
| 注意力头数 | `n_h` | 128 |
| 每头维度 | `c` | 512 |
| RoPE 维度 | `d_rope` | 64 |
| Q 低秩维度 | `d_c` | 1536 |
| O 分组数 / O 低秩维度 | `g` / `d_g` | 16 / 1024 |
| KV 压缩维度 | `d_kv` | 512 (FP8, 混合精度) |
| Indexer 头数/头维/topk | `n_I` / `d_I` / `k` | 64 / 128 / 1024 |
| SWA 窗口 | `n_win` | 128 |
| CSA 压缩率 | `m` | 4 (30 层) |
| HCA 压缩率 | `m'` | 128 (31 层) |
| MoE 路由专家/激活 | | 384 / 6 (+ 1 共享) |
| MoE 专家中间维度 | `d_ff` | 3072 |
| 激活参数 / 格式 | | 49B / FP8 |
| mHC 副本数 | `n_hc` | 4 |
| GPU FP8 算力 | | 2 PFLOPS |
| HBM 带宽 / 容量 | | 3.35 TB/s / 80 GB |
| PCIe 5.0 ×16 | | ~64 GB/s |
| NVLink-C2C | | ~900 GB/s |

---

## 二、全模型 FLOPs 分解

### 2.1 精确 FLOPs 公式

基于 [[raw/01-articles/zartbot-DeepSeek-V4详细分析：数据表格.md]] 的逐组件分解：

**单层 CSA（ratio=4）**:

| 组件 | FLOPs/token |
|------|-------------|
| 注意力投影 (`wq_a/b`, `wkv`, `wo_a/b`) | 599.7M |
| 主 Compressor (`wkv` + `wgate` + softmax + APE) | 29.4M |
| Indexer 固定部分 (`wq_b` + `weights_proj`) | 26.1M |
| Indexer Compressor | 7.3M |
| Indexer Score | **4,096 × Seq** |
| 注意力计算 (QK^T + AV, top-k=1024 + SWA=128) | 302.0M |
| **CSA 单层小计** | **4,096 × Seq + 964.5M** |

**单层 HCA（ratio=128）**:

| 组件 | FLOPs/token |
|------|-------------|
| 注意力投影 | 599.7M |
| 主 Compressor (`wkv` + `wgate`) | 14.7M |
| 注意力计算 (Dense: S/128 + SWA=128) | **2,048 × Seq + 33.54M** |
| **HCA 单层小计** | **2,048 × Seq + 647.94M** |

**全模型汇总**:

| 组件 | 层数 | FLOPs/token |
|------|------|-------------|
| LM Head | 1 | 1.85G |
| CSA 层 | 30 | 30 × (4,096 × Seq + 964.5M) = **122,880 × Seq + 28.93G** |
| HCA 层 | 31 | 31 × (2,048 × Seq + 647.94M) = **63,488 × Seq + 20.08G** |
| HC 连接 | 61 | 61 × 2.8M = **0.17G** |
| MoE FFN | 61 | 61 × 930.3M = **56.75G** |
| **总计** | | **186,368 × Seq + 107.77G** |

### 2.2 各序列长度总 FLOPs

| 序列长度 | V3.2 (GFLOPs) | V4-Pro (GFLOPs) | V4-Pro vs V3.2 | V4-Pro 计算时延 @2P |
|---------|---------------|-----------------|---------------|---------------------|
| **8k** | 91.72 | 109.26 | 0.84× (慢 19%) | **54.6 μs** |
| **32k** | 115.14 | 113.73 | 1.01× (持平) | **56.9 μs** |
| **128k** | 208.84 | 131.63 | 1.59× (快 59%) | **65.8 μs** |
| **256k** | 333.77 | 155.48 | 2.15× (快 115%) | **77.7 μs** |
| **512k** | 583.62 | 203.19 | 2.87× (快 187%) | **101.6 μs** |
| **1M** | 1083.33 | 298.61 | 3.63× (快 263%) | **149.3 μs** |

> **关键转折点**：在 32K 附近 V4-Pro 与 V3.2 的 FLOPs 基本持平；超过 32K 后 V4-Pro 的效率优势指数级放大。这是因为 V4-Pro 的注意力 FLOPs 系数仅为 186K/Seq，远小于 V3.2 的 O(Seq²) 注意力。

---

## 三、KV Cache 精确建模

### 3.1 V3.2 与 V4-Pro KV Cache 结构对比

基于技术报告与 [[KV Cache 管理]] 中的混合精度存储设计：

**V3.2 KV Cache 组成**（per layer）:

| 组件 | 精度 | 维度 | 每 token 字节 |
|------|------|------|-------------|
| `kv_cache` | FP8 | 512 | 512 B |
| `pe_cache` | BF16 | 64 | 128 B |
| `indexer_k_cache` | FP8 | 128 | 128 B |
| `indexer_k_scale` | FP32 | 1 | 4 B |
| **V3.2 合计** | | | **772 B/token/layer** |
| **V3.2 总计 (61层)** | | | **Seq × 61 × 772 B** |

**V4-Pro KV Cache 组成**（per layer）:

| 组件 | 精度 | CSA (30层) | HCA (31层) |
|------|------|-----------|-----------|
| SWA KV | FP8 | 128 × 512 B | 128 × 512 B |
| 压缩 KV | FP8 | Seq/4 × 512 B | Seq/128 × 512 B |
| Indexer K | FP4 | Seq/4 × 64 B | — |
| State Buffer | FP32 | 4×2×512×4 B | 128×1×512×4 B |
| Score Buffer | FP32 | 4×2×512×4 B | 128×1×512×4 B |

### 3.2 各序列长度 KV Cache 精确值

| 序列长度 | V3.2 | V4-Pro | 节省比 | HBM 占用率 (80GB) | 单卡并发数 |
|---------|------|--------|--------|-------------------|-----------|
| **8k** | 367.90 MB | **55.26 MB** | 6.6× | 0.07% | ~1,480 |
| **32k** | 1,471.62 MB | **159.42 MB** | 9.2× | 0.20% | ~513 |
| **128k** | 5,886.50 MB | **576.04 MB** | 10.2× | 0.72% | ~142 |
| **256k** | 11,773 MB | **1,131.54 MB** | 10.4× | 1.41% | ~72 |
| **512k** | 23,546 MB | **2,242.54 MB** | 10.5× | 2.80% | ~36 |
| **1M** | 47,092 MB | **4,464.54 MB** | 10.5× | 5.58% | ~18 |

> **首次修正**：此前建模将 1M KV Cache 错误估计为 3.9 GB（偏差 -12.5%），实际精确值为 **4.46 GB**。差异主要源于对 state buffer 和 MTP 层 KV 的低估。

---

## 四、Decoding 单步时延详细建模（batch=1）

### 4.1 HBM 内存读取量

基于 KV Cache 结构逐组件估算每 decode step 的 HBM 读取：

| 组件 | 读取策略 | 8k | 32k | 128k | 256k | 512k | 1M |
|------|---------|-----|------|------|------|------|-----|
| **模型权重** (49B FP8) | 全量读取 | 49 GB | 49 GB | 49 GB | 49 GB | 49 GB | 49 GB |
| **CSA Indexer 扫描** | 全量 (FP4, d_I=128) | 3.8 MB | 15.4 MB | 61.4 MB | 122.9 MB | 245.8 MB | **480 MB** |
| **CSA Top-k 加载** | k=1024 块 × 512B | 15.4 MB | 15.4 MB | 15.4 MB | 15.4 MB | 15.4 MB | 15.4 MB |
| **HCA Dense 加载** | S/128 × 512B × 31 | 1.0 MB | 3.9 MB | 15.5 MB | 31.0 MB | 62.0 MB | **124 MB** |
| **SWA 加载** | 128 × 512B × 61 | 3.9 MB | 3.9 MB | 3.9 MB | 3.9 MB | 3.9 MB | 3.9 MB |
| **KV 总读取** | | **~24 MB** | **~39 MB** | **~96 MB** | **~173 MB** | **~327 MB** | **~623 MB** |
| **总 HBM 读取** | | **~49.02 GB** | **~49.04 GB** | **~49.10 GB** | **~49.17 GB** | **~49.33 GB** | **~49.62 GB** |
| **内存时延 @3.35TB/s** | | **14.63 ms** | **14.64 ms** | **14.66 ms** | **14.68 ms** | **14.73 ms** | **14.81 ms** |

### 4.2 单 Token Decode 总时延（batch=1）

| 序列长度 | 计算时延 | 内存时延 | **实际瓶颈时延** | 瓶颈占比 |
|---------|---------|---------|-----------------|---------|
| **8k** | 54.6 μs | 14.63 ms | **14.63 ms** | 权重 99.6% |
| **32k** | 56.9 μs | 14.64 ms | **14.64 ms** | 权重 99.6% |
| **128k** | 65.8 μs | 14.66 ms | **14.66 ms** | 权重 99.5% |
| **256k** | 77.7 μs | 14.68 ms | **14.68 ms** | 权重 99.4% |
| **512k** | 101.6 μs | 14.73 ms | **14.73 ms** | 权重 99.3% |
| **1M** | 149.3 μs | 14.81 ms | **14.81 ms** | 权重 98.9% |

> **核心结论（修正后仍成立）**：Decoding 完全受限于**模型权重加载**（49 GB FP8，恒定 ~14.6 ms/step）。V4-Pro 的计算量从 8K 到 1M 仅增长 2.7×（109G→299G），而 V3.2 增长 11.8×（92G→1083G）。KV Cache 读取即使到 1M 也仅占 HBM 总读取的 1.3%（623 MB / 49.6 GB）。

---

## 五、H2D 预取带宽需求分析

### 5.1 场景定义与公式

- **连续批处理**：batch_i 计算时，同时从 DRAM 向 HBM 预取 batch_{i+1} 的 KV Cache
- **预取窗口** = batch_i 的单步 decode 时延（B=32 时 ≈ 14.6 ms）
- 采用精确 KV Cache 数值：$KV_{per\_req}(S)$ 取自第三节表格

$$H2D\_BW_{required} = \frac{(1 - hit\_rate) \times B \times KV_{per\_req}(S)}{T_{step}}$$

### 5.2 全量 KV 预取场景

| 序列长度 | Per-Request KV | Batch KV (B=32) | 70% 命中 (30% miss) | 90% 命中 (10% miss) | 95% 命中 (5% miss) | 99% 命中 (1% miss) |
|---------|---------------|-----------------|---------------------|---------------------|--------------------|--------------------|
| **8k** | 55.26 MB | 1.77 GB | 0.53 GB → **36.3 GB/s** ✅ | 0.18 GB → **12.1 GB/s** ✅ | 0.09 GB → **6.1 GB/s** ✅ | 0.02 GB → **1.2 GB/s** ✅ |
| **32k** | 159.42 MB | 5.10 GB | 1.53 GB → **104.8 GB/s** ⚠️ | 0.51 GB → **34.9 GB/s** ✅ | 0.26 GB → **17.5 GB/s** ✅ | 0.05 GB → **3.5 GB/s** ✅ |
| **128k** | 576.04 MB | 18.43 GB | 5.53 GB → **378.7 GB/s** ⚠️ | 1.84 GB → **126.2 GB/s** ⚠️ | 0.92 GB → **63.1 GB/s** ✅ | 0.18 GB → **12.6 GB/s** ✅ |
| **256k** | 1,131.54 MB | 36.21 GB | 10.86 GB → **744.0 GB/s** ❌ | 3.62 GB → **248.0 GB/s** ⚠️ | 1.81 GB → **124.0 GB/s** ⚠️ | 0.36 GB → **24.8 GB/s** ✅ |
| **512k** | 2,242.54 MB | 71.76 GB | 21.53 GB → **1,474.7 GB/s** ❌ | 7.18 GB → **491.6 GB/s** ❌ | 3.59 GB → **245.8 GB/s** ⚠️ | 0.72 GB → **49.2 GB/s** ✅ |
| **1M** | 4,464.54 MB | 142.87 GB | 42.86 GB → **2,935.8 GB/s** ❌ | 14.29 GB → **978.6 GB/s** ❌ | 7.14 GB → **489.3 GB/s** ❌ | 1.43 GB → **97.9 GB/s** ⚠️ |

**图例**：✅ PCIe 5.0 ×16 (~64 GB/s) 可达 | ⚠️ 需 NVLink-C2C (~900 GB/s) | ❌ 需 NVSwitch / 多链 NVLink

### 5.3 关键发现

1. **99% HBM 命中率依然是长上下文的硬门槛**：1M 上下文时，99% 命中率需求 ~98 GB/s，NVLink-C2C 刚好可达；若降至 95%，需求飙升至 489 GB/s，单卡 NVLink-C2C 也无法满足。

2. **短上下文（≤32k）极为宽容**：即使 70% 低命中率，H2D 需求仍 ≤105 GB/s，NVLink-C2C 全覆盖。

3. **HBM 容量约束**：80 GB HBM 最多容纳 ~18 个 1M 上下文的完整 KV Cache。Batch=32 时至少 14 个请求需要 DRAM→HBM 预取。

4. **与先前建模差异**：KV Cache 从 3.9 GB 修正为 4.46 GB 后，H2D 需求相应增加 ~14%，但趋势一致。

---

## 六、Indexer 与 Attention 分离：流水线化预取策略

### 6.1 分离可行性

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

**数据依赖分析**：
1. **Indexer 仅依赖**：当前 token 的 query + Indexer-KV cache（FP4, d_I=128, 480 MB @1M for all 30 CSA layers），全部驻留 HBM。**不消费** attention 输出。
2. **Attention 仅依赖**：Top-k 索引（Indexer 输出）+ 对应的完整 KV 块（FP8, d_kv=512）。
3. **无循环依赖**，天然可流水线化。

### 6.2 流水线预取时间轴

```
时间轴 →

Batch_i:     |── Indexer(i) ──|── KV Prefetch(i) ──|──── Attention(i) + FFN(i) ────|
Batch_{i+1}:                   |── Indexer(i+1) ──|── KV Prefetch(i+1) ──|── Attn(i+1) ...

重叠区域:     [═ Batch_i 的 Attention + FFN + MoE ═══════════╗
              ║ Batch_{i+1} 的 Indexer + Selective Prefetch  ═╝
```

| 时间点 | Batch_i 操作 | Batch_{i+1} 并行操作 | 数据量 |
|--------|-------------|---------------------|--------|
| T1 | CSA Attention (top-k) + MoE FFN | **Indexer QK Score** (FP4) 扫描 S/4 个压缩块 | ~480 MB (FP4 scan) |
| T2 | HCA Attention (dense) + MoE FFN | **Top-k 选择**：选出 score 最高的 1024 块 | — |
| T3 | Output Projection | **发起 H2D Prefetch**：仅预取 Top-1024 块的完整 KV | **15.4 MB (CSA) + S 比例 HCA** |

### 6.3 Top-k 预取量精确计算

**关键洞察**：CSA 层不需要预取全量压缩 KV，只需预取 Indexer 选出的 top-k=1024 个块。

每 CSA 层 top-k KV：$k \times d_{kv} \times 1\text{ byte (FP8)} = 1024 \times 512 = 512 \text{ KB}$
30 CSA 层：$30 \times 512 \text{ KB} = \mathbf{15.36 \text{ MB}}$（固定，与序列长度无关！）

| 序列长度 | 全量 KV Cache | CSA Full KV | CSA Top-k KV | HCA Full KV | **需预取总量** | **预取缩减比** |
|---------|-------------|------------|-------------|------------|--------------|--------------|
| **8k** | 55.26 MB | 30.7 MB | **15.4 MB** | 0.99 MB | **16.3 MB** | **3.4:1** |
| **32k** | 159.42 MB | 122.9 MB | **15.4 MB** | 3.88 MB | **19.2 MB** | **8.3:1** |
| **128k** | 576.04 MB | 491.5 MB | **15.4 MB** | 15.5 MB | **30.9 MB** | **18.6:1** |
| **256k** | 1,131.54 MB | 983.0 MB | **15.4 MB** | 31.0 MB | **46.4 MB** | **24.4:1** |
| **512k** | 2,242.54 MB | 1,966.1 MB | **15.4 MB** | 62.0 MB | **77.4 MB** | **29.0:1** |
| **1M** | 4,464.54 MB | 3,932.2 MB | **15.4 MB** | 124.0 MB | **139.4 MB** | **32.0:1** |

### 6.4 修正后的 H2D 带宽需求（Indexer 先行 Top-k 预取）

| 序列长度 | 需预取/req | B=32 总量 | 95% 命中 (5% miss) | 90% 命中 (10% miss) |
|---------|-----------|----------|--------------------|--------------------|
| **1M** | 139.4 MB | 4.46 GB | 5% × 4.46 / 14.6 ms = **15.3 GB/s** ✅ | 10% × 4.46 / 14.6 ms = **30.5 GB/s** ✅ |
| **512K** | 77.4 MB | 2.48 GB | **8.5 GB/s** ✅ | **17.0 GB/s** ✅ |
| **256K** | 46.4 MB | 1.48 GB | **5.1 GB/s** ✅ | **10.2 GB/s** ✅ |

> **核心结论**：采用 Indexer 先行的 top-k 预取策略后，**即使 90% HBM 命中率，1M 上下文的 H2D 需求也仅 30.5 GB/s**，轻松落在 PCIe 5.0 ×16（64 GB/s）范围之内。预取缩减比在 1M 时高达 32:1（从 4.46 GB 降至 139 MB/请求）。

### 6.5 为什么有效——架构视角

这依赖于 DeepSeek-V4 的三层注意力分工（[[Compressed Sparse Attention]] + [[Heavily Compressed Attention]]）：

- **HCA**：全局低分辨率概览，压缩率 128:1，1M 时全量 KV 仅 124 MB（31 层合计），**全量预取也不是负担**
- **CSA**：中等分辨率 + top-k=1024 稀疏选择，通过 Indexer 先确定"骨架边"再精确加载，**只需预取 15.4 MB**（30 层，与 Seq 无关）
- **SWA**：局部 128 token 细节，KV 可忽略不计（~4 MB，61 层合计）

Indexer 天然充当"**预取预测器**"：它用极小的 (FP4, 480 MB) 成本扫描全局，精确输出接下来注意力需要的 top-k 地址，使选择性预取成为可能。

---

## 七、总结

| 维度 | 修正前 | 修正后 | 结论变化 |
|------|--------|--------|---------|
| **单 Token FLOPs @1M** | ~32 GFLOPs | **298.6 GFLOPs** | 低估 9.3×，MoE FFN (56.75G) 是主要差异来源 |
| **KV Cache @1M** | ~3.9 GB | **4.46 GB** | 低估 12.5%，State Buffer 和 MTP 层此前未计入 |
| **计算瓶颈** | 权重加载 (~14.6 ms) | 权重加载 (~14.6 ms) | **结论不变**，但计算占比从 0.08% 升至 1.0% |
| **H2D 全量预取 @1M/99%** | 85.5 GB/s | **97.9 GB/s** | NVLink-C2C 仍然够用，但裕量收窄 |
| **H2D Top-k 预取 @1M/90%** | 40.1 GB/s | **30.5 GB/s** | PCIe 5.0 完全够用，裕量更足 |
| **Indexer-Attention 分离** | ✅ 可行 | ✅ 可行 | **确认**：Indexer FP4 全扫描 480 MB + Top-k 精确预取 15.4 MB 的流水线策略有效 |

---

## 关联连接
- [[DeepSeek-V4]] — 主体模型架构
- [[DeepSeek-V4-Pro]] — 建模目标规格（7168/61层/30CSA+31HCA）
- [[Compressed Sparse Attention]] — CSA 4:1 压缩 + Indexer 稀疏选择
- [[Heavily Compressed Attention]] — HCA 128:1 压缩 + Dense Attention
- [[Lightning Indexer]] — FP4 Indexer，天然充当预取预测器
- [[KV Cache 管理]] — 混合精度存储（BF16 RoPE + FP8 KV + FP4 Indexer）
- [[MegaMoE]] — MoE 计算-通信重叠
- [[FP4 量化感知训练]] — Indexer FP4 QAT
- [[raw/01-articles/zartbot-DeepSeek-V4详细分析：数据表格.md]] — 精确 FLOPs 与 KV Cache 数据来源
