'''
Overview:
* The code builds an MDP-based RL agent by creating an environment with a grid and obstacles, defining actions, and implementing Q-learning to optimize the agent's policy through experience. 
* It also uses value iteration to derive an optimal policy based on dynamic programming. 
* Finally, it benchmarks the efficiency and effectiveness of both methods by measuring execution time and comparing the derived policies.
'''
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from collections import deque

# Parameters
grid_size = 100
episodes = 100
alpha = 0.1
gamma = 0.9
epsilon = 0.1
decay_rate = 0.99  # For decaying epsilon

actions = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1)
}

# Initialize grid and Q-table
grid = np.zeros((grid_size, grid_size))
start = (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
end = (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
Q = np.zeros((grid_size, grid_size, len(actions)))

# Place obstacles randomly on the grid
num_obstacles = 3000
for _ in range(num_obstacles):
    x, y = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
    if (x, y) != start and (x, y) != end:
        grid[x][y] = -1

def is_valid_move(x, y):
    return 0 <= x < grid_size and 0 <= y < grid_size and grid[x][y] != -1

def get_reward(x, y):
    if (x, y) == end:
        return 10
    elif grid[x][y] == -1:
        return -10
    else:
        return -1

def q_learning(episodes=100, max_steps=500):
    global epsilon
    rewards_per_episode = []
    for _ in range(episodes):
        x, y = start
        total_reward = 0
        steps = 0
        while (x, y) != end and steps < max_steps:
            if random.uniform(0, 1) < epsilon:
                action = random.choice(list(actions.keys()))
            else:
                action = max(actions, key=lambda a: Q[x, y, list(actions.keys()).index(a)])
            dx, dy = actions[action]
            nx, ny = x + dx, y + dy
            if is_valid_move(nx, ny):
                reward = get_reward(nx, ny)
                total_reward += reward
                old_q = Q[x, y, list(actions.keys()).index(action)]
                max_next_q = np.max(Q[nx, ny])
                Q[x, y, list(actions.keys()).index(action)] = old_q + alpha * (reward + gamma * max_next_q - old_q)
                x, y = nx, ny
            steps += 1
        rewards_per_episode.append(total_reward)
        epsilon *= decay_rate  # Decaying epsilon
    return rewards_per_episode

# Q-learning execution and timing
start_time = time.time()
rewards_per_episode = q_learning(episodes=episodes)
q_learning_time = time.time() - start_time

plt.plot(range(episodes), rewards_per_episode)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Improved Q-Learning Performance on 100x100 Grid")
plt.show()

theta = 0.1
V = np.zeros((grid_size, grid_size))  # State-value function

def value_iteration():
    delta = float('inf')
    while delta > theta:
        delta = 0
        for x in range(grid_size):
            for y in range(grid_size):
                if (x, y) == end or grid[x][y] == -1:
                    continue
                v = V[x, y]
                V[x, y] = max([get_reward(x + dx, y + dy) + gamma * V[x + dx, y + dy]
                               if is_valid_move(x + dx, y + dy) else -10
                               for (dx, dy) in actions.values()])
                delta = max(delta, abs(v - V[x, y]))
    # Derive policy from value function
    policy = np.full((grid_size, grid_size), "", dtype=object)
    for x in range(grid_size):
        for y in range(grid_size):
            if (x, y) == end or grid[x][y] == -1:
                continue
            best_action = max(actions, key=lambda a: get_reward(x + actions[a][0], y + actions[a][1]) +
                                                gamma * V[x + actions[a][0], y + actions[a][1]]
                                                if is_valid_move(x + actions[a][0], y + actions[a][1]) else -10)
            policy[x, y] = best_action
    return policy

start_time = time.time()
dp_policy = value_iteration()
value_iteration_time = time.time() - start_time

# Output time and policies
print("Q-Learning Time:", q_learning_time)
print("Value Iteration Time:", value_iteration_time)
print("\nDP Policy derived from Value Iteration:")
print(dp_policy)

def get_q_learning_policy(Q):
    q_policy = np.full((grid_size, grid_size), "", dtype=object)
    for x in range(grid_size):
        for y in range(grid_size):
            if (x, y) == end or grid[x][y] == -1:
                continue
            best_action = max(actions, key=lambda a: Q[x, y, list(actions.keys()).index(a)])
            q_policy[x, y] = best_action
    return q_policy

q_policy = get_q_learning_policy(Q)

print("\nQ-Learning Derived Policy:")
print(q_policy)

'''
Sample Output:
Q-Learning Time: 0.22255635261535645
Value Iteration Time: 2.504316806793213

DP Policy derived from Value Iteration:
[['down' 'down' 'down' ... '' 'right' 'left']
 ['up' 'up' 'up' ... 'left' '' 'up']
 ['' 'up' 'up' ... 'left' 'left' '']
 ...
 ['up' 'up' 'up' ... 'up' '' '']
 ['up' '' '' ... '' 'down' 'left']
 ['up' 'left' 'left' ... 'right' 'up' 'up']]

Q-Learning Derived Policy:
[['up' 'up' 'up' ... '' 'up' 'up']
 ['up' 'up' 'up' ... 'up' '' 'up']
 ['' 'up' 'up' ... 'up' 'up' '']
 ...
 ['up' 'up' 'up' ... 'up' '' '']
 ['up' '' '' ... '' 'up' 'up']
 ['up' 'up' 'up' ... 'up' 'up' 'up']]
'''
