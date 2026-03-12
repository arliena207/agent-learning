# cartpole-dqn

这个目录是 CartPole 的调库 DQN 实验最小实现，包含训练脚本和实验笔记。

## 文件说明

- `dqn_cartpole.py`：使用 `gymnasium + stable-baselines3` 训练 `CartPole-v1` 的 DQN 脚本。
- `Cartpole-DQN-experiment.md`：本实验的分析笔记（MDP 结构、reward 含义、gamma/lr 影响等）。
- `requirements.txt`：项目依赖清单（统一安装入口）。
- `readme.md`：当前说明文件。

## 环境准备

```bash
python3 -m venv rl_env
source rl_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## requirements.txt 说明

`requirements.txt` 只保留项目直接依赖：

- `gymnasium[classic-control]`：提供 `CartPole-v1` 环境。
- `stable-baselines3`：提供 DQN 算法实现。

使用 `pip install -r requirements.txt` 后，`stable-baselines3` 会自动安装其所需的底层依赖（如 `torch`、`numpy` 等）

## 运行方式

在当前目录执行：

```bash
python dqn_cartpole.py
```

脚本当前配置：

- 环境：`CartPole-v1`
- 算法：`DQN("MlpPolicy", env, exploration_initial_eps=0.8, verbose=1)`
- 训练步数：`total_timesteps=100000`

## 当前脚本行为

- 会直接开始训练并在终端输出训练日志。
- 目前没有保存模型（`model.save`）和单独评估流程。

## 可选改进

1. 训练后保存模型参数，便于复现和对比。
2. 增加评估脚本，统计固定回合平均 reward。
3. 做 `gamma`、`learning rate` 的对比实验并记录曲线。
