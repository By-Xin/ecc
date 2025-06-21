import gym
import numpy as np
import math
import random
from scipy.stats import mannwhitneyu

# 离散化参数
NUM_BINS = 10
p_bound    = np.linspace(-2.4,  2.4,  NUM_BINS - 1)
v_bound    = np.linspace(-3.0,  3.0,  NUM_BINS - 1)
ang_bound  = np.linspace(-0.5,  0.5,  NUM_BINS - 1)
angv_bound = np.linspace(-2.0,  2.0,  NUM_BINS - 1)
BOUNDS     = [p_bound, v_bound, ang_bound, angv_bound]

UCB_C      = 1.0
LEARNING_RATE = 0.5
DISCOUNT      = 0.9
EPISODES      = 10000
WINDOW        = 1000
N_RUNS        = 15

def discretize_state(state):
    """把连续状态离散化到 0..NUM_BINS-1"""
    return tuple(
        np.digitize(s, b)
        for s, b in zip(state, BOUNDS)
    )

def select_action_ucb(q_table, count_table, ds):
    """基于 UCB 的行动选择"""
    counts_s     = count_table[ds]
    total_counts = counts_s.sum() or 1
    ucb_values   = np.zeros_like(counts_s, dtype=float)
    for a in range(len(counts_s)):
        if counts_s[a] == 0:
            ucb_values[a] = np.inf
        else:
            bonus = UCB_C * math.sqrt(math.log(total_counts) / counts_s[a])
            ucb_values[a] = q_table[ds + (a,)] + bonus
    action = int(np.argmax(ucb_values))
    count_table[ds + (action,)] += 1
    return action

def run_experiment(reward_type):
    """
    运行一次实验，返回每个 episode 的长度列表。
    reward_type: 'dense' 或 'sparse'
    """
    env = gym.make('CartPole-v1')
    action_dim = env.action_space.n

    # 初始化 Q 表 和 计数表
    q_table     = np.zeros((NUM_BINS,)*4 + (action_dim,), dtype=float)
    count_table = np.zeros_like(q_table, dtype=int)

    lengths = []
    for epi in range(EPISODES):
        state, _ = env.reset()
        ds = discretize_state(state)
        step = 0
        done = False

        while not done:
            step += 1
            # 选动作
            action = select_action_ucb(q_table, count_table, ds)

            # 执行动作
            nxt, reward_env, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            nds = discretize_state(nxt)

            # 计算自定义奖励
            if reward_type == 'dense':
                reward = reward_env  # CartPole 中每步 reward_env=1
            elif reward_type == 'sparse':
                # 提前失败惩罚 -1，成功坚持到 400 步以上给 +1，其它步 0
                if done and step < 400:
                    reward = -1
                elif done:
                    reward = 1
                else:
                    reward = 0
            else:
                raise ValueError("reward_type 必须是 'dense' 或 'sparse'")

            # Q-learning 更新
            current_q = q_table[ds + (action,)]
            # 如果终止，则没有后续价值
            max_next = 0 if done else np.max(q_table[nds])
            td_target = reward + DISCOUNT * max_next
            q_table[ds + (action,)] += LEARNING_RATE * (td_target - current_q)

            ds = nds

        lengths.append(step)

    env.close()
    return np.array(lengths)

def summarize_runs(lengths_array):
    """
    lengths_array: shape (N_RUNS, EPISODES)
    返回每次 run 在最后 WINDOW 个 episode 上的平均步长
    """
    return np.array([
        lengths_array[i, -WINDOW:].mean()
        for i in range(lengths_array.shape[0])
    ])

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)

    # 分别跑 N_RUNS 次两种奖励方式
    dense_final = np.zeros(N_RUNS, dtype=float)
    sparse_final = np.zeros(N_RUNS, dtype=float)

    for i in range(N_RUNS):
        print(f'Running dense reward experiment {i+1}/{N_RUNS}...')
        lengths = run_experiment(reward_type='dense')
        dense_final[i] = lengths[-WINDOW:].mean()

        print(f'Running sparse reward experiment {i+1}/{N_RUNS}...')
        lengths = run_experiment(reward_type='sparse')
        sparse_final[i] = lengths[-WINDOW:].mean()

    # 输出两组结果
    print("\nDense reward results (mean last 1000 episodes):")
    print(dense_final)
    print("\nSparse reward results (mean last 1000 episodes):")
    print(sparse_final)

    # Wilcoxon rank-sum test (Mann–Whitney U)
    stat, p_value = mannwhitneyu(dense_final, sparse_final, alternative='two-sided')
    print("\nWilcoxon rank-sum test (Mann–Whitney U):")
    print(f"  U statistic = {stat:.2f}")
    print(f"  p-value      = {p_value:.4f}")

    if p_value < 0.05:
        print("结果显著：两种奖励方式表现有统计学差异。")
    else:
        print("结果不显著：无法拒绝两种奖励方式表现相同的原假设。")