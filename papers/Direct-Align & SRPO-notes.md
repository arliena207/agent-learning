# Direct-Align & SRPO 论文阅读笔记

### 论文标题

**Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference**

### 论文链接

- arXiv 页面：https://arxiv.org/abs/2509.06942
- PDF 链接：https://arxiv.org/pdf/2509.06942

### 论文作者

Xiangwei Shen, Zhimin Li, Zhantao Yang, Shiyi Zhang, Yingfang Zhang,Donghao Li, Chunyu Wang, Qinglin Lu, Yansong Tang

## 1. 研究背景与挑战

这篇论文关注的是 text-to-image diffusion model 的 human preference alignment。已有工作证明，可以通过 differentiable reward 直接对 diffusion model 做对齐，让生成结果更符合人类偏好。

核心问题：

1. **reward计算依赖多步去噪和梯度传播，优化成本高**

2. **reward 机制不灵活，依赖离线训练，难以动态控制**

## 2. 核心工作

核心工作分为两部分：

- Direct-Align：从diffusion优化机制上解决多步 diffusion 成本高、后期 timestep 优化偏向的问题。
- SRPO（Semantic Relative Preference Optimization）：从 reward 设计上解决 reward 偏好难动态控制的问题。

构建了一个 Online-RL learning framework for text-to-image generation：

```text
diffusion noisy state
→ Direct-Align 单步恢复图像 / 多timestep reward聚合
→ SRPO 构造文本条件化、相对偏好的reward
→ Online-RL 优化 diffusion model
```

## 3. Direct-Align
- 利用 diffusion 中间状态是 图像与噪声插值 这一性质，从任意 timestep 直接恢复图像，从而避免完整的多步 trajectory 优化。

    - 传统 diffusion：$x_t \rightarrow x_{t-1} \rightarrow x_{t-2} \rightarrow \cdots \rightarrow x_0$

    - Direct-Align 改写：$x_t \rightarrow x_0$

- reward
    - design：单步恢复图像后计算 reward
        $$
        r = r\left(
            \frac{
            x_t - \Delta\sigma_t \, \epsilon_\theta(x_t, t, c)
            - (\sigma_t - \Delta\sigma)\,\epsilon
            }{\alpha_t}
            \right)
        $$

    - aggregation：多个 timestep 的 reward 共同参与优化

        $$
        r(x_t) = \lambda(t) \cdot \sum_{i=k-n}^{k}
        r\big(x_i - \epsilon_\theta(x_i, i, c), c\big)
        $$

## 4. Semantic Relative Preference Optimization (SRPO)：

SRPO 从 reward 设计角度改进 diffusion alignment，将 reward 从绝对评分转为文本条件化 + 相对偏好优化。

- SGP
在 prompt 里加入控制词，改变文本 embedding，从而调节 reward 的语义方向，实现动态控制：

    $$
    r_{\text{SGP}}(\mathbf{x}) = RM(\mathbf{x}, (p_c, p)) 
    \propto f_{\text{img}}(\mathbf{x})^T \cdot C_{(p_c, p)}
    $$

- SRP
将 reward 从绝对评分转化为正负偏好差：

    $$
    r_{\text{SRP}}(\mathbf{x}) = r_1 - r_2
    = f_{\text{img}}(\mathbf{x})^T \cdot (C_1 - C_2)
    $$

- Inversion-Based Regularization

在 denoising 和 inversion 两个方向分别进行梯度优化，将奖励项与惩罚项在不同 timestep 上解耦。


## 5. 总结
这篇论文的核心贡献是把 diffusion alignment 拆成了两个更本质的问题：如何高效地覆盖完整 trajectory，以及 如何用更可控的方式表达偏好。
对应地，Direct-Align 从优化机制上解决了多步 denoising 过于昂贵、只能优化后期 timestep 的问题；SRPO 则从 reward 设计上，将静态绝对评分改成了文本条件化、相对偏好的优化信号。

整体上，这篇工作**把 diffusion 的 human preference alignment 从局部、静态的 reward 优化推进到了全 trajectory、可控偏好的联合对齐**。


## 6. 个人思考
我觉得这篇论文最有价值的地方在于，它没有把问题局限在换一个更强的 reward model，而是把 diffusion 优化路径 和 reward 偏好表达两个层面一起重构了。

其中 Direct-Align 说明：对 diffusion 来说，训练效率本身就是 alignment 能力的一部分；SRPO 则说明：偏好不一定要表示成单一分数，也可以表示成带语义方向的相对信号。这个思路让 alignment 从被动打分，变成了更主动的、可结构化控制的过程。