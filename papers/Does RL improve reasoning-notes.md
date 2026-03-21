# Does RL improve reasoning 论文阅读笔记

## 重要信息

### 论文标题

**Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?**

### 论文链接

- arXiv 页面：https://arxiv.org/abs/2504.13837
- PDF 链接：https://arxiv.org/pdf/2504.13837

### 论文作者

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, Gao Huang

## 1. 研究背景与核心问题

RLVR（Reinforcement Learning with Verifiable Rewards）在数学、代码和视觉推理任务上显著提升了小样本指标（如 pass@1）。  
关键区分：这种提升是否代表**推理能力边界（reasoning boundary）扩展**，还是仅代表**采样效率提升（sampling efficiency）**。

## 2. 主要思路

作者使用大样本条件下的 `pass@k` 来近似评估能力边界，并进行跨模型、跨 RL 算法、跨任务（数学/代码/视觉推理）的系统对比。

- 若 RLVR 扩展了能力边界，则在大 `k` 下应覆盖更多 base model 原本无法解出的样本。
- 若 RLVR 主要提升采样效率，则优势应集中在小 `k`，大 `k` 不一定超过 base model。

## 3. 核心结果

1. RLVR 在小 `k`（尤其 `pass@1`）上普遍优于 base model。  
2. 在大 `k` 条件下，base model 往往达到更高上界，说明 RLVR 主要改善了“命中已有正确路径”的效率。  
3. `coverage` 与 `perplexity` 分析显示，RLVR 生成路径多数仍落在 base model 原有分布内。  
4. 与 RLVR 相比，distillation 更可能引入新的推理模式，因而更有机会扩大能力边界。

## 4. 方法与结论的局限

指标提升、采样效率提升、能力边界扩展是三个不同层次。  
当前 RLVR 更接近前两者，不足以证明“通过 RL 学到全新推理策略”。


## 5. 总结

这篇论文的贡献不在于提出新算法，而在于给出一个更严格的问题：  
**RL 提升了什么？是推理策略本身，还是策略采样效率？**

当前证据支持的结论是：现有 RLVR 主要提升采样效率，而非显著扩展推理能力边界。

## 6. 我的思考

这篇论文最有价值的启发是，评估推理模型时，应显式区分性能提升和能力提升。

- 不能只看 `pass@1` 或单点 benchmark，需要结合大 `k` 指标分析能力上界。  
- 对 RL Agent 或多步决策研究而言，关键不只是 reward 优化本身，还包括探索机制、过程监督与环境交互设计。
