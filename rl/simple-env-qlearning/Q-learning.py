import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class Config():
    N_STATES = 8
    START_STATE = 0
    GOAL_STATE = 6
    TRAP_STATE = 7

    A_LEFT = 0
    A_RIGHT = 1
    ACTIONS = [A_LEFT, A_RIGHT]

    EPSILON = 0.2
    EPISODES = 80
    MAX_STEPS = 30
    RISK_SLIP_PROB = 0.35

STATE_NAME = {
    0: "START",
    1: "SAFE_1",
    2: "RISK_1",
    3: "SAFE_2",
    4: "RISK_2",
    5: "SAFE_3",
    6: "GOAL",
    7: "TRAP",
}

# 每个状态下动作对应的下一状态（确定性转移）
TRANSITIONS = {
    0: {Config.A_LEFT: 1, Config.A_RIGHT: 2},
    1: {Config.A_LEFT: 3, Config.A_RIGHT: 3},
    2: {Config.A_LEFT: 4, Config.A_RIGHT: 4},
    3: {Config.A_LEFT: 5, Config.A_RIGHT: 5},
    4: {Config.A_LEFT: 7, Config.A_RIGHT: 6},
    5: {Config.A_LEFT: 6, Config.A_RIGHT: 6},
    6: {Config.A_LEFT: 6, Config.A_RIGHT: 6},
    7: {Config.A_LEFT: 7, Config.A_RIGHT: 7},
}

def step(state, action):
    # 风险分支存在随机性：在 RISK_2 执行向右时，有概率滑入 TRAP
    if state == 4 and action == Config.A_RIGHT:
        if np.random.random() < Config.RISK_SLIP_PROB:
            next_state = Config.TRAP_STATE
        else:
            next_state = Config.GOAL_STATE
    else:
        next_state = TRANSITIONS[state][action]

    if next_state == Config.GOAL_STATE:
        return next_state, 5.0, True

    if next_state == Config.TRAP_STATE:
        return next_state, -5.0, True

    # 风险分支中间状态给即时正奖励，形成“短期收益 vs 长期稳定”的权衡
    if next_state in (2, 4):
        return next_state, 1.0, False
    
    return next_state, 0.0, False

def choose_action(state, q_table, epsilon):
    if random.random() < epsilon:
        # 探索
        return np.random.choice(Config.ACTIONS)
    # 利用
    return np.argmax(q_table[state])
    

def train(gamma, alpha, episodes, epsilon, seed=42, max_steps=Config.MAX_STEPS):
    random.seed(seed)
    np.random.seed(seed)

    q_table = np.zeros((Config.N_STATES, len(Config.ACTIONS)))
    episode_returns = []

    for _ in range(episodes):
        state = Config.START_STATE
        done = False
        ep_return = 0.0
        steps = 0

        while not done and steps < max_steps:
            action = choose_action(state, q_table, epsilon)
            next_state, reward, done = step(state, action)

            best_next = np.max(q_table[next_state])
            target = reward + gamma * best_next
            td_error = target - q_table[state][action]
            q_table[state][action] += alpha * td_error

            ep_return += reward
            state = next_state
            steps += 1

        episode_returns.append(ep_return)

    return q_table, episode_returns


def evaluate_greedy(q_table, eval_episodes=200, seed=123, max_steps=Config.MAX_STEPS):
    random.seed(seed)
    np.random.seed(seed)

    total_return = 0.0
    success = 0

    for _ in range(eval_episodes):
        state = Config.START_STATE
        done = False
        ep_return = 0.0
        steps = 0

        while not done and steps < max_steps:
            action = int(np.argmax(q_table[state]))  # 纯贪心
            next_state, reward, done = step(state, action)
            ep_return += reward
            state = next_state
            steps += 1

        total_return += ep_return
        if state == Config.GOAL_STATE:
            success += 1

    avg_return = total_return / eval_episodes
    success_rate = success / eval_episodes
    return avg_return, success_rate

def print_greedy_policy(q_table):
    action_name = {Config.A_LEFT: "L", Config.A_RIGHT: "R"}
    print("\nGreedy policy:")
    for s in range(Config.N_STATES):
        if s in (Config.GOAL_STATE, Config.TRAP_STATE):
            print(f"state {s} ({STATE_NAME[s]}): terminal")
            continue
        a = int(np.argmax(q_table[s]))
        print(f"state {s} ({STATE_NAME[s]}): {action_name[a]}   Q={np.round(q_table[s], 3)}")

def plot_trends(results):
    gammas = sorted(list({item["gamma"] for item in results}))
    alphas = sorted(list({item["alpha"] for item in results}))

    # 按 gamma 画 alpha 变化趋势
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4))
    for gamma in gammas:
        subset = [x for x in results if x["gamma"] == gamma]
        subset = sorted(subset, key=lambda x: x["alpha"])
        x = [s["alpha"] for s in subset]
        y_return = [s["avg_return"] for s in subset]
        y_success = [s["success_rate"] for s in subset]
        axes1[0].plot(x, y_return, marker="o", label=f"gamma={gamma}")
        axes1[1].plot(x, y_success, marker="o", label=f"gamma={gamma}")

    axes1[0].set_title("Avg Return vs Alpha")
    axes1[0].set_xlabel("alpha")
    axes1[0].set_ylabel("avg_return")
    axes1[0].grid(True, alpha=0.3)
    axes1[0].legend()

    axes1[1].set_title("Success Rate vs Alpha")
    axes1[1].set_xlabel("alpha")
    axes1[1].set_ylabel("success_rate")
    axes1[1].grid(True, alpha=0.3)
    axes1[1].legend()

    fig1.tight_layout()
    fig1.savefig("gamma_fixed_alpha_trends.png", dpi=150)

    # 按 alpha 画 gamma 变化趋势
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    for alpha in alphas:
        subset = [x for x in results if x["alpha"] == alpha]
        subset = sorted(subset, key=lambda x: x["gamma"])
        x = [s["gamma"] for s in subset]
        y_return = [s["avg_return"] for s in subset]
        y_success = [s["success_rate"] for s in subset]
        axes2[0].plot(x, y_return, marker="o", label=f"alpha={alpha}")
        axes2[1].plot(x, y_success, marker="o", label=f"alpha={alpha}")

    axes2[0].set_title("Avg Return vs Gamma")
    axes2[0].set_xlabel("gamma")
    axes2[0].set_ylabel("avg_return")
    axes2[0].grid(True, alpha=0.3)
    axes2[0].legend()

    axes2[1].set_title("Success Rate vs Gamma")
    axes2[1].set_xlabel("gamma")
    axes2[1].set_ylabel("success_rate")
    axes2[1].grid(True, alpha=0.3)
    axes2[1].legend()

    fig2.tight_layout()
    fig2.savefig("alpha_fixed_gamma_trends.png", dpi=150)
    plt.close(fig1)
    plt.close(fig2)

if __name__=="__main__":
    gammas = [0.1, 0.3, 0.5, 0.7, 0.9]
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    seeds = [0, 1, 2, 3, 4]
    
    print("gamma  alpha  avg_return  success_rate")
    best = None
    results = []

    for gamma in gammas:
        for alpha in alphas:
            metrics = []
            last_q_table = None
            for seed in seeds:
                q_table, _ = train(
                    gamma=gamma,
                    alpha=alpha,
                    episodes=Config.EPISODES,
                    epsilon=Config.EPSILON,
                    seed=seed,
                )
                avg_return, success_rate = evaluate_greedy(q_table, eval_episodes=200, seed=seed + 1000)
                metrics.append((avg_return, success_rate))
                last_q_table = q_table

            mean_return = float(np.mean([m[0] for m in metrics]))
            mean_success = float(np.mean([m[1] for m in metrics]))
            print(f"{gamma:<5}  {alpha:<5}  {mean_return:<10.3f}  {mean_success:<.3f}")
            results.append(
                {
                    "gamma": gamma,
                    "alpha": alpha,
                    "avg_return": mean_return,
                    "success_rate": mean_success,
                }
            )

            if best is None or mean_success > best["success_rate"]:
                best = {"gamma": gamma, "alpha": alpha, "Q": last_q_table, "success_rate": mean_success}

    print("\nBest setting:")
    print(f"gamma={best['gamma']}, alpha={best['alpha']}, success_rate={best['success_rate']:.3f}")
    print_greedy_policy(best["Q"])
    plot_trends(results)
    print("\nSaved plots:")
    print("- gamma_fixed_alpha_trends.png")
    print("- alpha_fixed_gamma_trends.png")
