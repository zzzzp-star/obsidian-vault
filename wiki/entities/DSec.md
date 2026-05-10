---
title: "DSec"
type: entity
tags: [沙箱, Agent, 基础设施, DeepSeek, Rust]
sources: [raw/01-articles/DeepSeek-V4技术报告解读-从架构到 Infra 的全栈重构.md]
last_updated: 2026-05-10
---

## 定义
DeepSeek Elastic Compute（DSec）是 DeepSeek-V4 的 Agent 训练底座——一套 Rust 编写的沙箱平台，由 API gateway、per-host agent、cluster monitor 三个组件组成，跑在 3FS 分布式文件系统上。单集群管理数十万并发 sandbox 实例。

## 关键信息

### 四级执行 Substrate
| 类型 | 用途 | 技术栈 |
|------|------|--------|
| Function Call | 无状态轻量调用，消除冷启动 | 自研容器池 |
| Container | 完整 Docker 兼容 | EROFS on-demand loading |
| microVM | VM 级隔离，安全敏感 | Firecracker |
| fullVM | 支持任意 Guest OS | QEMU |

四级共享相同 API（命令执行、文件传输、TTY 访问），参数切换即可。

### Fast Image Loading
- Container base image 和文件系统 commit 存为 3FS-backed 只读 EROFS 层
- 元数据本地 mount，数据块按需拉
- microVM 用 overlaybd，基层只读共享、写走 copy-on-write，链式快照毫秒级恢复

### Trajectory Logging 三用
1. **Client fast-forwarding**：训练抢占时保留沙箱资源，恢复时 replay 已完成命令
2. **Fine-grained provenance**：每个状态变化可追溯来源
3. **Deterministic replay**：任意历史 session 可从 trajectory 完整复现

### Agent 训练四大支柱
1. **Execution Authenticity**：真实命令、网络调用、文件系统操作
2. **Reward Accessibility**：rule-based verifier + rubric-GRM
3. **Trajectory Reproducibility**：全序 log + 毫秒级快照
4. **Capability Mergeability**：OPD 蒸馏回统一模型

## 关联连接
- [[DeepSeek-V4]] — 主体模型
- [[Specialist训练范式]] — 训练范式
- [[Generative Reward Model]] — GRM 评分
- [[On-Policy Distillation]] — 能力融合
