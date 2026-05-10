# 操作日志

## [2026-05-10] ingest | 批量摄入 DeepSeek-V4 相关知识库（8 个源文件）
- **变更**: 
  - 新增 Sources (8): [[摘要-deepseek-v4-zartbot-analysis]], [[摘要-deepseek-v4-full-stack-review]], [[摘要-deepseek-v4-rope-design]], [[摘要-llm-inference-parallel-strategies]], [[摘要-engram-paper]], [[摘要-mhc-paper]], [[摘要-deepseek-v4-paper]], [[摘要-deepseek-v3-2-paper]]
  - 新增 Entities (5): [[DeepSeek-V4]], [[DeepSeek-V4-Pro]], [[DeepSeek-V4-Flash]], [[DeepSeek-V3.2]], [[DSec]]
  - 新增 Concepts (18): [[Compressed Sparse Attention]], [[Heavily Compressed Attention]], [[Manifold-Constrained Hyper-Connections]], [[Muon Optimizer]], [[大模型推理并行策略]], [[On-Policy Distillation]], [[Generative Reward Model]], [[DeepSeekMoE]], [[MegaMoE]], [[Lightning Indexer]], [[FP4 量化感知训练]], [[Multi-Head Latent Attention]], [[Specialist训练范式]], [[Sinkhorn-Knopp 算法]], [[Stiefel 流形]], [[Multi-Token Prediction]], [[Newton-Schulz 迭代]], [[RoPE]], [[KV Cache 管理]]
  - 更新 [[index.md]]：建立完整知识目录
- **源文件**: 
  - raw/01-articles/zartbot-DeepSeek-V4详细分析（1）：算法和模型结构.md
  - raw/01-articles/DeepSeek-V4技术报告解读-从架构到 Infra 的全栈重构.md
  - raw/01-articles/DeepSeekV4中RoPE设计解析.md
  - raw/01-articles/大模型推理并行策略(DP,TP,PP,SP,EP)原理简介.md
  - raw/02-papers/Engram_paper.pdf
  - raw/02-papers/mHC.pdf
  - raw/02-papers/DeepSeek_V4.pdf
  - raw/02-papers/DeepSeek_V3_2.pdf
- **冲突**: 无（Wiki 初始构建）

## [2026-05-10] query | DeepSeek-V4 基于 2 PFLOPS GPU 的时延与带宽建模分析
- **输出**: 综合 [[DeepSeek-V4]]、[[DeepSeek-V4-Pro]]、[[Compressed Sparse Attention]]、[[Heavily Compressed Attention]]、[[Lightning Indexer]]、[[KV Cache 管理]] 的分析
- **内容**: (1) 6 种序列长度下的 KV Cache 大小与解码时延详细建模；(2) 基于 70%/90%/95%/99% HBM 命中率的 H2D 预取带宽需求推算；(3) Indexer-Attention 分离可行性分析与 top-k 预取优化策略
- **关键发现**: Decoding 完全受限于权重加载（~14.6ms/step）；Indexer 可充当 KV 预取预测器，将 CSA 预取量从 GB 级降至 0.8 MB；采用 Indexer 先行的 top-k 预取 + 90% HBM 命中率即可在 PCIe 5.0 带宽内完成 1M 上下文的 KV 预取

## [2026-05-10] query | 保存 synthesis 页面
- **变更**: 新增 [[synthesis-deepseek-v4-latency-bandwidth-model]]; 更新 [[index.md]]
- **冲突**: 无
