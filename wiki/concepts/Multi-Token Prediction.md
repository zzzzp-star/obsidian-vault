---
title: "Multi-Token Prediction"
type: concept
tags: [模型架构, 预测, DeepSeek]
sources: [raw/01-articles/zartbot-DeepSeek-V4详细分析（1）：算法和模型结构.md]
last_updated: 2026-05-10
---

## 定义
多 Token 预测（Multi-Token Prediction, MTP）是 DeepSeek 系列模型从 V3 延续下来的模块，在标准的下一个 Token 预测之外，模型同时预测后续多个 Token。DeepSeek-V4 继承了 V3 的 MTP 配置，未做修改。

## 关键信息

### 设计目的
- 提升训练信号密度：每个位置预测多个 token，增加监督信号
- 提升推理效率：可一次生成多个 token
- 已被 V3 验证有效，V4 直接沿用

### V4 中的应用
- MTP 深度：1（预测 1 个额外 token）
- 在 MTP 层使用滑动窗口 KV Cache（128 token, FP8）
- MTP 层不参与 CSA/HCA 压缩

## 关联连接
- [[DeepSeek-V4]] — 应用模型
- [[DeepSeek-V3.2]] — 首次引入 MTP
- [[DeepSeekMoE]] — 配合使用的 MoE 架构
