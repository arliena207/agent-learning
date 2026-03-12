### Agent笔记

#### RL和监督学习的区别

·RL数据具有时间连续性，是作为一个序列，而监督学习的数据一般比较独立

·RL的rewards具有时间滞后性，而监督学习有即时反馈

·RL做出的行动会影响他下次得到的数据，即会改变环境

·RL一直在做探索与试错

#### Agent与Environment

·Agent—————action—————\>Environment

·Agent\<—————reward—————Environment

#### RL Agent组成部分

**1.策略（policy）：agent的表现行为**

·stochastic
policy：$`\pi(a|s) = P\lbrack A_{t} = a \mid S_{t} = s\rbrack`$

·determinisitic policy：$`a^{\ast} = \arg{\max_{a}\pi}(a|s)`$

**2.价值函数（value
function）：环境/行动的评估（折合因子**$`\mathbf{y}`$**）**

·中值函数（状态值函数）：从状态$`s`$出发能获得的未来折扣奖励的期望总和\
``` math
v_{\pi}(s) \doteq \mathbb{E}_{\pi}\left\lbrack G_{t} \mid S_{t} = s \right\rbrack = \mathbb{E}_{\pi}\left\lbrack \sum_{k = 0}^{\infty}\gamma^{k}R_{t + k + 1\ \ }|\ S_{t} = s \right\rbrack,\quad for\ all\ s \in \mathcal{S}
```

·Q函数（动作值函数）：从状态$`s`$出发，执行动作$`a`$，能获得的未来折扣奖励的期望总和\
``` math
q_{\pi}(s,a) \doteq \mathbb{E}_{\pi}\left\lbrack G_{t} \mid S_{t} = s,A_{t} = a \right\rbrack = \mathbb{E}_{\pi}\left\lbrack \sum_{k = 0}^{\infty}\gamma^{k}R_{t + k + 1}\ |\ S_{t} = s,A_{t} = a \right\rbrack
```

其中，return是$`G_{t} = \sum_{k = 0}^{\infty}\gamma^{k}R_{t + k + 1\ \ }`$（从现在开始未来所有奖励的加权总和）

**3.模型（model）：agent的对环境状态的整体理解**

·下个状态：$`\mathcal{P}_{ss'}^{a} = \mathbb{P}\left\lbrack S_{t + 1} = s' \mid S_{t} = s,A_{t} = a \right\rbrack`$（状态转移概率）

·下一个奖励：$`\mathcal{R}_{s}^{a} = \mathbb{E}\left\lbrack R_{t + 1} \mid S_{t} = s,A_{t} = a \right\rbrack`$（期望奖励）

#### Exploration and Exploitation

Exploration：探索新事物，可能会让Agent变好，但也可能探索错误

Exploitation：采用已经验证最好的策略，可能会陷入局部最优

#### MDPS

1.  **五元tuple：**$`\mathbf{（}\mathbf{S}\mathbf{，}\mathbf{A}\mathbf{，}\mathbf{P}\mathbf{，}\mathbf{R}\mathbf{，}\mathbf{\gamma}`$**）**

    ·$`\mathbf{S}`$（state）：状态空间

    ·$`\mathbf{A}`$（action）：动作空间

    ·$`\mathbf{P}`$（transition probability）：如果我在状态 s 采取动作
    a，下一个状态变成 s’ 的概率是多少？

    ·$`\mathbf{R}`$（Reward）：奖励函数

    ·$`\mathbf{\gamma}`$（gamma）：折扣因子

2.  **Bellman Expectation Equation**

    ·Bellman state-value function Equation：\
    ``` math
    v^{\pi}(s) = \mathbb{E}_{\pi}\left\lbrack R_{t + 1} + \gamma v^{\pi}(s_{t + 1}) \mid s_{t} = s \right\rbrack = \sum_{a \in \mathcal{A}}^{}\pi(a|s)\left( R(s,a) + \gamma\sum_{s' \in \mathcal{S}}^{}P(s'|s,a)v^{\pi}(s') \right)
    ```

    ·Bellman action-value function Equation：

    ``` math
    q^{\pi}(s,a) = \mathbb{E}_{\pi}\left\lbrack R_{t + 1} + \gamma q^{\pi}(s_{t + 1},A_{t + 1}) \mid s_{t} = s,A_{t} = a \right\rbrack = R(s,a) + \gamma\sum_{s' \in \mathcal{S}}^{}P(s'|s,a)\sum_{a' \in \mathcal{A}}^{}\pi(a'|s')q^{\pi}(s',a')
    ```

    Policy Evaluation：对于Bellman expectation
    backup反复递归，直至收敛：

    ``` math
    v_{t + 1}(s) = \sum_{a \in \mathcal{A}}^{}\pi(a|s)\left( R(s,a) + \gamma\sum_{s' \in \mathcal{S}}^{}P(s'|s,a)v_{t}(s') \right)
    ```

    解释：猜一个初始$`v_{t}(s)`$，不断用Bellman去更新直到稳定

3.  为什么Return要取期望？

    Return 是随机的。因为环境有随机性（P 是概率），policy
    可能是随机的。所以同一个状态 s，未来奖励可能不同。

    Return期望意义在于：在当前 policy
    下，从这个状态出发，平均能拿多少回报。
