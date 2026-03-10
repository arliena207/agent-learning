import numpy as np
import random

N_STATES=5
GOAL_STATE=N_STATES-1
A_LEFT=0
A_RIGHT=1
ACTIONS=[A_LEFT,A_RIGHT]
GAMMA=0.5
ALPHA=0.5

Q = np.zeros((5,2))

def step(state, action):
    if action == A_LEFT:
        next_state = state-1 if state-1 >= 0 else 0
    else:
        next_state = state+1 if state+1 <= N_STATES-1 else N_STATES-1

    if next_state == GOAL_STATE:
        return next_state, 1.0, True

    if next_state == 2:
        return next_state,-1.0,True
    
    return next_state, 0.0, False

epsilon = 0.2
def choose_action(state):
    if random.random() < epsilon:
        # 探索
        return np.random.choice(ACTIONS)
    else:
        # 利用
        return np.argmax(Q[state])
    

episodes = 100
for ep in range(episodes):
    state = 0
    done = False

    while not done:
        action = choose_action(state)
        next_state, reward, done = step(state, action)

        # Q ,state 更新
        best_next=np.max(Q[next_state])
        target = reward + GAMMA*best_next
        error = target-Q[state][action]
        Q[state,action] += ALPHA * error
        state = next_state

print(np.round(Q,3))
