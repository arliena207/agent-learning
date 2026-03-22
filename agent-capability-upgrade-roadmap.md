# Agent 能力升级路线图

## 总路线图

RL决策基础 → 深度RL结构理解 → 感知+决策融合 → Transformer决策建模 → 多模态Agent成型

---

## 一、路线图总览

### Phase 2：深度 RL 结构理解

目标：从 tabular Q-learning 升级到 DQN / PPO 的结构级理解

### Phase 3：感知 + 决策融合

目标：做第一个真正像 Agent 的项目（Vision + RL）

### Phase 4：Transformer 决策建模

目标：把 RL 和序列建模接起来，进入 Decision Transformer / LLM Agent 视角

### Phase 5：多模态 Agent 成型

目标：形成你的主线研究雏形：多模态输入、内部状态、推理决策、动作输出

### Phase 6：实习 / 研究输出层

目标：把学习成果变成 GitHub、技术总结、项目展示、简历素材

---

## 二、阶段总览表

| 阶段 | 时间建议 | 核心问题 | 产出 |
|---|---|---|---|
| Phase 2 | 3-6周 | 深度RL为什么有效/不稳定 | DQN from scratch、PPO理解 |
| Phase 3 | 4-6周 | 感知如何接决策 | Vision-RL Agent Demo |
| Phase 4 | 3-5周 | Transformer如何做决策 | Mini Transformer、Decision Transformer Demo |
| Phase 5 | 4-8周 | 多模态Agent闭环怎么搭 | 多模态Agent原型 |
| Phase 6 | 持续 | 如何对外展示 | GitHub、README、总结、简历 |

---

## 五、Phase 2：深度RL结构理解

### Phase 2 的核心目标

你要真正讲清 4 个问题：
1. 为什么 DQN 比 tabular Q-learning 更难训练
2. replay buffer 为什么有效
3. target network 为什么有效
4. policy-based 方法为什么会出现

---

### Phase 2-2：手写 DQN from scratch

类型：手搓

这是你第二阶段最重要的项目。

仓库名称建议：

`dqn-from-scratch`

你要自己写什么：
- `q_network.py`：MLP Q 网络
- `replay_buffer.py`：经验回放池
- `agent.py`：epsilon-greedy、update逻辑
- `train.py`：训练循环
- `utils.py`：日志/画图

核心算法结构（你要自己实现）：
- Q network
- TD target
- replay sampling
- target network update
- epsilon-greedy

不要自己手写的部分：
- CartPole 环境本身
- PyTorch autograd
- 可视化可以用 matplotlib / tensorboard

要记录的指标：
- episode reward
- loss
- mean Q-value

你要做的对比实验：
- gamma：0.95 vs 0.99
- learning rate：5e-5 vs 1e-4 vs 5e-4
- target update frequency：慢/快对比

---

### Phase 2-3：理解 DQN 系列改进

你不一定全实现，但一定要知道它们解决什么问题。

1. Double DQN  
核心问题：Q-value overestimation

2. Dueling DQN  
核心问题：state value 和 action advantage 分开建模

3. Prioritized Experience Replay  
核心问题：经验采样更高效

你至少要能做一个对照表：

| 算法 | 解决问题 | 核心思想 |
|---|---|---|
| DQN | 高维状态的 value learning | NN 近似 Q |
| Double DQN | 过估计 | 选择动作和估值分开 |
| Dueling DQN | 表达效率 | V 和 A 分开建模 |
| PER | 样本效率 | 高 TD-error 样本优先采样 |

---

### Phase 2-4：读 Policy Gradient / PPO

当你理解 DQN 后，就该补 policy-based 这条线。

先读哪篇：

PPO 原始论文：

**Proximal Policy Optimization Algorithms**

链接：

https://arxiv.org/abs/1707.06347

你这一步要重点理解：
- policy vs value
- on-policy vs off-policy
- advantage 是什么
- clipping 是什么
- entropy bonus 是什么

你现在不用马上手写 PPO 全版。

先做到：
- 能看懂 PPO 基本更新式
- 能解释它为什么比 vanilla policy gradient 稳定

---

### Phase 2 的项目与输出

项目 1：`dqn-from-scratch`

输出 1：
一篇 DQN 总结，标题可以是：

**从 Q-learning 到 DQN：为什么要 replay 和 target network**

输出 2：
一个 10 分钟口头讲解稿，内容包括：
- DQN pipeline
- Bellman backup
- replay buffer
- target network

---

### Phase 2 完成标准

你应该能：
- 讲清 DQN 机制
- 讲清 policy vs value
- 讲清 on-policy / off-policy
- 讲清 PPO 在解决什么问题
- 有一个 DQN from scratch 项目仓库

这时候你就已经不是“只会跑课程作业”的状态了。

---

## 六、Phase 3：感知 + 决策融合（真正开始像 Agent）

这一阶段开始从“纯RL”走向“Agent”。

核心思想：

决策系统不应该只接受表格状态，  
它应该能从视觉输入中抽象状态，再做决策。

### Phase 3 的核心项目：Vision-RL Agent

类型：半手搓，半调库

项目名称建议：`vision-rl-agent`

项目目标：
让 Agent 从图像输入中做动作。

结构：

```text
Image
↓
CNN encoder
↓
state embedding
↓
policy / Q network
↓
action
```

环境建议（优先级从高到低）：
1. MiniGrid（最推荐）
2. 简单 Atari
3. 自己写的小 2D grid world with image observation

你要做什么：
- 用 CNN 做感知编码
- 用 DQN / PPO 做决策
- 训练一个能自动行动的 Agent

你要自己写到什么程度：

手搓部分：
- CNN 编码器
- 决策头（Q 或 policy）
- 训练循环（如果你能力允许）

调库部分：
- 环境用现成
- PPO / DQN 优化器逻辑可调库
- 自己重点写 perception pipeline

最终展示形式：
- 输入游戏画面
- Agent 自动玩
- 输出 reward 曲线
- README 讲清 perception → action pipeline

### 这个阶段你会真正理解什么

- 感知不是决策
- 图像不是状态本身，需要 encoder
- RL 在这里学的是“基于表示的决策”

### Phase 3 的论文

1. Chain-of-Visual Reasoning  
链接：http://arxiv.org/abs/2511.19418

2. Dual-Process Reasoning for Large Models  
链接：http://arxiv.org/abs/2506.01955

### Phase 3 完成标准

你应该有：
- 一个 Vision + RL 的 demo
- 一个技术报告，讲 perception → policy → action
- 能解释 Agent pipeline

---

## 七、Phase 4：Transformer 决策建模

这是你从 RL 走向 LLM / Agent 的桥梁阶段。

核心思想：

决策不一定非要是 value function，  
也可以是 sequence modeling。

### Phase 4-1：手写 mini Transformer

类型：手搓

仓库名称建议：`mini-transformer`

你要实现什么：
- embedding
- self-attention
- multi-head attention
- residual
- layernorm
- MLP block

你不需要实现什么：
- 大规模训练
- 大模型优化技巧
- tokenizer 复杂工程

你的目标：
- self-attention 在做什么
- residual / layernorm 为什么稳定
- Transformer 为什么适合长序列建模

### Phase 4-2：Decision Transformer Demo

类型：半手搓

论文：

**Decision Transformer: Reinforcement Learning via Sequence Modeling**

链接：

https://arxiv.org/abs/2106.01345

核心思想：

把 trajectory 当成序列：
`return-to-go, state, action, state, action...`

交给 Transformer 学习。

你要做什么：
- 小环境（MiniGrid / 简化 gridworld）
- 收集 trajectory
- 用小 Transformer 预测 action

这个项目的意义：

这是 RL 和 LLM 的真正桥梁。你会第一次感受到：  
原来“决策”可以写成“序列预测”。

### Phase 4 完成标准

你应该有：
- mini Transformer 项目
- Decision Transformer demo
- 一篇总结：为什么 sequence modeling 可以做决策

---

## 八、Phase 5：多模态 Agent 成型

这是你的主线研究雏形阶段。

这里不要求你做成工业级系统，要求是你能做一个完整闭环原型。

### Phase 5 的目标

构建一个最小多模态 Agent：

```text
Vision / Language input
↓
internal representation
↓
reasoning / planning
↓
action / edit / tool call
```

### 可选路线一：Vision + Language + Decision

输入：
- 图像
- 文本指令

输出：
- 编辑动作 / 环境动作

示例：
“把图中左上角物体移动到中间”

Agent 需要：
- 视觉定位
- 指令理解
- 规划操作
- 执行动作

### 可选路线二：Tool-Use Agent（简化版）

结构：

```text
LLM
↓
生成 action / tool call
↓
tool execution
↓
feedback
```

这个路线更偏 LLM Agent，不是你现在最优先，但可以做个简单版 demo。

### 可选路线三：图像编辑 Agent（贴近你师兄方向）

这个方向最对口。

你可以做一个简化版：

输入：
- 图像
- 编辑目标（文本）

中间：
- reasoning / planning
- 可能调用一个 image editing model

输出：
- 编辑动作序列 / 编辑后结果

这个阶段不要求你自己发明新算法，  
要求你能把“视觉理解 + 决策 + 编辑执行”串起来。

### 这阶段要补的论文

按优先级：

精读：
- ThinkRL：http://arxiv.org/abs/2601.03467
- Does Reinforcement Learning Improve Reasoning?：http://arxiv.org/abs/2504.13837
- Chain-of-Visual Reasoning：http://arxiv.org/abs/2511.19418
- Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference：http://arxiv.org/abs/2509.06942

半精读：
- Can We Improve LLM Reasoning via RL?：http://arxiv.org/abs/2501.13926
- Dual-Process Reasoning for Large Models：http://arxiv.org/abs/2506.01955

略读：
- The Promise of Reinforcement Learning for Reasoning：http://arxiv.org/abs/2508.01119
- Diffusion Model Alignment：http://arxiv.org/abs/2502.01051
- DiffusionNFT：http://arxiv.org/abs/2509.16117
- Video Generation with Diffusion Models：http://arxiv.org/abs/2511.21541
- EditThinker：http://arxiv.org/abs/2512.05965
- EditScore：http://arxiv.org/abs/2509.23909

### Phase 5 完成标准

你应该有一个完整 demo，能讲清：
- 感知模块
- 状态表示
- 决策模块
- 执行模块
- 整体 pipeline

这时候你就真的站在“Agent 研究”的门口了。

---

## 九、Phase 6：输出层（GitHub / 总结 / 简历）

这一阶段其实是全程并行做的。

你最终至少要落成这几个项目：

1. 项目 1：Q-learning from scratch（手搓，环境：FrozenLake / 自定义环境）
2. 项目 2：DQN from scratch（手搓，环境：CartPole）
3. 项目 3：Vision-RL Agent（半手搓，环境：MiniGrid / 图像环境）
4. 项目 4：Mini Transformer（手搓）
5. 项目 5：Decision Transformer Demo（半手搓）
6. 项目 6（可选）：简化图像编辑 Agent / Tool-use Agent（原型级）

每个仓库都应该有：
- README
- requirements
- train script
- 结果图
- 一段你自己的总结

---

## 十、按时间排成现实版本

结合你前面的时间线，排成现实版：

### 现在 – 3月中

- 完成 CartPole + DQN 实验报告
- 读 DQN 论文
- 开始手写 DQN
- 精读 4 篇师兄给的论文中的前 2 篇

### 3月中 – 5月底

- 完成 DQN from scratch
- 理解 Double DQN / Dueling / PER
- 读 PPO 论文
- 开始 mini Transformer
- 继续论文精读 + 半精读

### 6月（期末）

- 不学新算法
- 整理 GitHub
- 写“RL 从零到 DQN”总结

### 7月 – 8月中

- 做 Vision + RL 项目
- 或做 Decision Transformer demo
- 和暑研内容尽量对齐

### 8月中 – 10月

- 手写简化 PPO
- 整理融合项目
- 优化简历
- 准备投实习

### 11月

你应该拥有：
- Q-learning
- DQN
- PPO
- Vision-RL 项目
- Transformer/Decision Transformer Demo
- 多篇技术总结

---

## 技术能力

- RL 机制理解
- 深度RL结构理解
- Transformer 结构理解
- 感知+决策融合能力
- 多模态Agent基础能力

## 可展示成果

- GitHub 仓库
- 论文阅读报告
- 技术总结
- Demo 项目

---

## 当前完成进度判断（可选附录）

- Phase 0：已完成大部分（仓库结构和基础环境已搭好）
- Phase 1：已完成（Q-learning + CartPole DQN + 实验报告 + DQN论文预热）
- Phase 2：已启动（开始进入 DQN 结构理解与后续手写阶段）
