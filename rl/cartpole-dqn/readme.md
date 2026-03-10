# CartPole-DQN 学习项目

这个目录包含两个层次的强化学习实践：

1. 手写一个极简离散环境上的 `Q-learning`（用于理解 Bellman 更新与探索策略）。
2. 使用 `Stable-Baselines3` 在 `CartPole-v1` 上训练 `DQN`（用于实际环境训练）。

## 目录结构

- `Q-learning.py`：手写 Q-learning 示例（5 个状态、2 个动作的 toy 环境）。
- `dpn_cartpole.py`：基于 `gymnasium + stable-baselines3` 的 CartPole DQN 训练脚本。
- `Agent笔记.md`：RL 基础概念笔记（Policy、Value Function、MDP、Bellman 等）。
- `CartPole实验报告.md`：针对 `gamma` 和 `learning rate` 的实验理解与总结。

## 环境准备

```bash
python3 -m venv rl_env
source rl_env/bin/activate
pip install --upgrade pip
pip install gymnasium stable-baselines3
```

## 快速运行

### 1) 运行手写 Q-learning

```bash
python Q-learning.py
```

运行后会打印训练结束的 Q 表（`5 x 2`），用于观察每个状态下“向左/向右”动作价值。

### 2) 运行 CartPole DQN

```bash
python dpn_cartpole.py
```

当前脚本会：

- 创建 `CartPole-v1` 环境；
- 使用 `MlpPolicy` 初始化 DQN；
- 设置 `exploration_initial_eps=0.8`；
- 训练 `total_timesteps=100000`。

## 代码核心思路

### Q-learning.py（手写版本）

- 动作选择：`epsilon-greedy`（`epsilon = 0.2`）。
- 更新规则：

```text
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
```

- 关键参数：
  - `GAMMA = 0.5`：折扣因子（短期/长期回报权衡）。
  - `ALPHA = 0.5`：学习率（更新步长）。
  - `episodes = 100`：训练轮数。

### dpn_cartpole.py（SB3 DQN）

- 使用 DQN 在连续状态空间下学习离散动作策略。
- `exploration_initial_eps=0.8` 表示初始探索比例较高，有利于早期采样多样性。

## 参数理解（结合实验笔记）

- `gamma` 越大，模型越重视长期回报，通常更稳定但收敛更慢。
- `learning rate` 越大，学习更快但更容易震荡；越小，更新更平滑但训练更慢。


## 建议的下一步优化

1. 在 `dpn_cartpole.py` 增加 `model.save(...)`，保存训练模型。
2. 增加评估脚本（固定若干回合平均 reward）。
3. 对比不同 `gamma/lr` 组合并记录曲线。
4. 修正文件名 `dpn_cartpole.py` 为 `dqn_cartpole.py`（更符合语义，避免后续混淆）。
