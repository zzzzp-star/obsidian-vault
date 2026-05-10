# Wiki Index

## Sources
- [[摘要-deepseek-v4-zartbot-analysis]] — DeepSeek-V4 算法和模型结构的深度技术分析，含范畴论视角
- [[摘要-deepseek-v4-full-stack-review]] — DeepSeek-V4 从架构到 Infra 的全栈重构解读
- [[摘要-deepseek-v4-rope-design]] — DeepSeek-V4 中 RoPE 位置编码的设计选择解析
- [[摘要-llm-inference-parallel-strategies]] — 大模型推理并行策略（DP/TP/PP/SP/EP/CP）入门
- [[摘要-engram-paper]] — Engram 键值记忆论文概要
- [[摘要-mhc-paper]] — mHC 流形约束超连接论文概要
- [[摘要-deepseek-v4-paper]] — DeepSeek-V4 官方技术报告概要
- [[摘要-deepseek-v3-2-paper]] — DeepSeek-V3.2 技术报告概要

## Entities
- [[DeepSeek-V4]] — DeepSeek-V4 系列大语言模型总览，1M token 上下文高效智能
- [[DeepSeek-V4-Pro]] — V4 高配版，1.6T/49B，与 Claude Opus 4.6 持平
- [[DeepSeek-V4-Flash]] — V4 轻量版，284B/13B，极致性价比
- [[DeepSeek-V3.2]] — V4 的前代模型，MLA + DSA + MTP 架构
- [[DSec]] — DeepSeek 弹性计算沙箱平台，Agent 训练基础设施

## Concepts
### 注意力机制
- [[Compressed Sparse Attention]] — CSA 压缩稀疏注意力（ratio=4，压缩+稀疏选择）
- [[Heavily Compressed Attention]] — HCA 重度压缩注意力（ratio=128，压缩+密集注意力）
- [[Lightning Indexer]] — CSA 的稀疏选择索引器（低秩、低精度）
- [[Multi-Head Latent Attention]] — MLA 多头潜在注意力（V2/V3 前身）
- [[RoPE]] — 旋转位置编码，V4 CSA/HCA 中的位置编码设计

### 模型架构
- [[Manifold-Constrained Hyper-Connections]] — mHC 流形约束超连接，替代传统残差连接
- [[DeepSeekMoE]] — DeepSeek 混合专家架构（路由专家+共享专家）
- [[Multi-Token Prediction]] — MTP 多 Token 预测模块
- [[KV Cache 管理]] — KV 缓存的分级存储、混合精度与磁盘复用

### 优化器与训练
- [[Muon Optimizer]] — 矩阵级优化器，Newton-Schulz 正交化替代 AdamW
- [[Newton-Schulz 迭代]] — 矩阵极分解正交部分的高效数值算法
- [[Stiefel 流形]] — Muon 优化器的几何理论基础
- [[Sinkhorn-Knopp 算法]] — mHC 中用于双随机矩阵投影的迭代算法

### 后训练范式
- [[Specialist训练范式]] — 领域专家独立训练 + OPD 蒸馏整合
- [[On-Policy Distillation]] — 在线策略蒸馏，full-vocabulary reverse KL
- [[Generative Reward Model]] — 生成式奖励模型（Actor-as-GRM），处理难验证任务

### 基础设施
- [[MegaMoE]] — 超融合 EP kernel，计算与通信完全重叠
- [[FP4 量化感知训练]] — FP4 QAT，MoE 权重与 Indexer 低精度训练
- [[大模型推理并行策略]] — DP、TP、PP、SP、EP、CP 六种并行策略总览

## Syntheses
- [[synthesis-deepseek-v4-latency-bandwidth-model]] — DeepSeek-V4 基于 2 PFLOPS GPU 的推理时延与 H2D 带宽建模分析，含 Indexer-Attention 分离预取策略
