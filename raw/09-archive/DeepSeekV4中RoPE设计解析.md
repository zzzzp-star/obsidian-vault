---
title: DeepSeekV4中RoPE设计解析
source: https://zhuanlan.zhihu.com/p/2029134107490132079
author:
  - "[[kaiyuan​新知答主]]"
published:
created: 2026-05-10
description: DeepSeek V4采用RoPE进行位置编码，但由于注意力结构升级，会带来两个核心问题： 在CSA/HCA中存在压缩操作，多个token会被压缩为一个token，位置信息应在压缩前注入还是压缩后注入？Attention采用MQA模式，K与V共…
tags:
  - 模型架构
  - 注意力机制
  - DeepSeek
  - LLM
  - MoE
---
[收录于 · 深度学习基础知识](https://www.zhihu.com/column/c_1977663450219046624)

99 人赞同了该文章

目录

收起

1 MLA中的RoPE处理回顾

2 CSA/HCA中的RoPE处理

2.1 为什么要对输出O做一次旋转？

2.2 能否直接给P旋转？

2.3 旋转应在压缩前还是压缩后？

DeepSeek V4采用RoPE进行位置编码，但由于注意力结构升级，会带来两个核心问题：

1. 在CSA/ [HCA](https://zhida.zhihu.com/search?content_id=273369799&content_type=Article&match_order=1&q=HCA&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3Nzg1MjUyMzcsInEiOiJIQ0EiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNzMzNjk3OTksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.GXFXTzjj8EsUOIznXUoGh5VHlrenTSw-Fk3V8-7Vplc&zhida_source=entity) 中存在压缩操作，多个token会被压缩为一个token，位置信息应在压缩前注入还是压缩后注入？
2. Attention采用 [MQA](https://zhida.zhihu.com/search?content_id=273369799&content_type=Article&match_order=1&q=MQA&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3Nzg1MjUyMzcsInEiOiJNUUEiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNzMzNjk3OTksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.fRDy64nzvH-FbogrjooQg9ZZ9ZotkEAlDdslUHwsqo4&zhida_source=entity) 模式，K与V共享表示。若直接对KV旋转，会将位置信息引入V，该如何处理？

下面围绕这两个问题，梳理DSV4的位置编码设计。

## 1 MLA中的RoPE处理回顾

在分析V4之前，先回顾V2/V3的MLA（ **M** ulti-head **L** atent **A** ttention）方案，因为MLA同样涉及MQA与KV cache压缩问题。

在标准RoPE中，计算query（ $q_{m}$ ）与key（ $k_{n}$ ）内积时，可写为：

$\left(\right. R_{m} q_{m} \left.\right)^{\top} \left(\right. R_{n} k_{n} \left.\right) = q_{m}^{\top} R_{m}^{\top} R_{n} k_{n} = q_{m}^{\top} R_{n - m} k_{n}$

其中 $R_{m}$ 、 $R_{n}$ 是位置 $m$ 、 $n$ 对应的旋转矩阵。由于旋转矩阵正交，满足：

$R \left(\right. \theta \left.\right)^{\top} = R \left(\right. \theta \left.\right)^{- 1} = R \left(\right. - \theta \left.\right)$

因此内积只依赖相对位置 $n - m$ 。具体介绍参考《 [彻底搞懂RoPE计算原理](https://zhuanlan.zhihu.com/p/2023493768003724514) 》 [^1]

在MLA中，KV下采样后K与V共享同一份cache值，这样可以节省显存；但也带来问题：如果给K注入RoPE，V会被一并旋转，导致V值“掺杂”了位置信息。

一种直观做法是 **将K、V拆开，仅对K旋转** 。但这样需要分别存储K cache和V cache，开销会回到接近GQA。

MLA采用了一个折中的做法：在Q、K隐藏维度中设置一部分专门用于RoPE计算。

![](https://pica.zhimg.com/v2-2d382acb86c27289941095f9179f44bc_1440w.jpg)

> 图片参考：https://github.com/CalvinXKY/InfraTech/tree/main/models/deepseek\_v3

这样既能让K携带位置信息，又能避免污染V；同时只需额外存储较小的RoPE相关K cache（如图中的k\_pe），远小于完整拆分K/V cache。公式推导参考： [part2](https://link.zhihu.com/?target=https%3A//spaces.ac.cn/archives/10091) 、3 [^2]

## 2 CSA/HCA中的RoPE处理

在DSV4的CSA/HCA中，同样存在KV cache压缩与MQA下KV共享的问题。CSA与HCA在RoPE处理上的思路一致，下面以HCA为例说明。

HCA中涉及RoPE的主要位置包括：

1. 窗口通道（SWA）的KV值；
2. C128A压缩器输出的压缩KV值；
3. 上采样后的Q值；
4. Attention输出的O值。
![](https://pic4.zhimg.com/v2-7e976a17f80a8f550a9b876529f947c1_1440w.jpg)

> 图片参考：https://github.com/CalvinXKY/InfraTech/blob/main/models/deepseek\_v4

看到这个设计，思考问题如下：

### 2.1 为什么要对输出O做一次旋转？

前面在MLA中提到，KV共享时直接旋转KV会让V也带上位置信息。HCA中窗口通道与压缩通道都在KV的最后 `rope_head_dim` 维度上施加RoPE。对应维度上的计算可写为：

这里多出的等价于给输出 **引入绝对位置信息** 。

绝对位置信息并非一定不可训练，但在 **可扩展性** （尤其是长上下文外推）上通常不如相对位置形式稳定。

因此HCA会对输出再做一次逆旋转：

这样位置项从绝对位置转为 **相对位置** 。

这里有个小问题：是否能采用正向旋转 ？答案是否定的，因为从公式看到，结果将仍偏向绝对位置表达。

### 2.2 能否直接给P旋转？

不行。设末两维为 `[seq,head_dim]` ，而维度为 `[seq,seq]` 。RoPE旋转作用在 `head_dim` 维，与旋转维度不匹配。 从计算上看，可理解为“标量权重乘向量”，本身是标量权重集合，不具备可旋转的向量维度。

### 2.3 旋转应在压缩前还是压缩后？

RoPE角度与绝对位置相关：

其中是token位置索引，d为注意力头维度，即hidden\_size/num\_heads，i的取值范围是。而C128A会将128个KV状态压缩为1个KV状态，QK计算使用的是压缩后的K。

![](https://pic2.zhimg.com/v2-cc6a69bb37a9b2c88556525a150d32c7_1440w.jpg)

核心问题是：K旋转角度的系数位置m怎么选？

- 若在压缩前旋转：每个token先带位置再压缩。这样看似直观，但位置信息会在序列维累加混合，容易破坏RoPE所需的相对位置结构。
- 若在压缩后旋转：给每个压缩K值指定一个标定位置即可。该位置可选起始、结束或中点，只要映射规则全程一致。

HCA采用的是“每128段取起始位置”，旋转角度公式：

其中是当前压缩K值在压缩序列中的索引。

![](https://pic3.zhimg.com/v2-47a94e5a666d517f13bebfb4b60aa984_1440w.jpg)

C128A压缩计算中RoPE的位置

建议的前置阅读：

---

> 想深耕AI Infra领域？欢迎访问 [InfraTech](https://link.zhihu.com/?target=https%3A//github.com/CalvinXKY/InfraTech) 库！内容涵盖大模型基础、PyTorch/vLLM/SGLang框架入门、性能加速等核心方向，配套 [50+知识干货](https://link.zhihu.com/?target=https%3A//github.com/CalvinXKY/InfraTech) 及适合初学者的 [notebook](https://link.zhihu.com/?target=https%3A//github.com/CalvinXKY/InfraTech) 练习。

**文中不足之处**

## 参考

编辑于 2026-05-07 22:33・浙江

[^1]: [https://zhuanlan.zhihu.com/p/2023493768003724514](https://zhuanlan.zhihu.com/p/2023493768003724514)

[^2]: [https://spaces.ac.cn/archives/10091](https://spaces.ac.cn/archives/10091)