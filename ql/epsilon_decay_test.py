import numpy as np
import gym
import math
import matplotlib.pyplot as plt

# 环境与超参
env = gym.make("CartPole-v1")
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 30000  # 用于测试
N_RUNS = 5       # 每种策略运行次数

# 离散化参数
Observation = [30, 30, 50, 50]
np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1])

def get_discrete_state(state):
    discrete = state / np_array_win_size + np.array([15, 10, 1, 10])
    return tuple(discrete.astype(int))

# 不同的 epsilon 递减函数

def epsilon_exponential(initial, decay_rate, episode):
    return max(0.05, initial * (decay_rate ** episode))

def epsilon_linear(initial, min_epsilon, max_episodes, episode):
    slope = (initial - min_epsilon) / max_episodes
    return max(min_epsilon, initial - slope * episode)

def epsilon_inverse_time(initial, k, episode):
    return max(0.05, initial / (1 + k * episode))

# Q-learning 主循环，接收 epsilon 函数

def run_q_learning(epsilon_func, func_kwargs):
    # 初始化 Q 表
    q_table = np.random.uniform(0, 1, size=(Observation + [env.action_space.n]))
    episode_rewards = []

    for episode in range(EPISODES + 1):
        state, _ = env.reset()
        discrete_state = get_discrete_state(state)
        done = False
        episode_reward = 0

        # 计算当前 epsilon
        epsilon = epsilon_func(**func_kwargs, episode=episode)

        while not done:
            # 选择动作
            if np.random.random() > epsilon:
                action = np.argmax(q_table[discrete_state])
            else:
                action = np.random.randint(env.action_space.n)

            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            new_discrete_state = get_discrete_state(new_state)

            if not done:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q    = q_table[discrete_state + (action,)]
                new_q = (1 - LEARNING_RATE) * current_q + \
                       LEARNING_RATE * (reward + DISCOUNT * max_future_q)
                q_table[discrete_state + (action,)] = new_q

            discrete_state = new_discrete_state

        episode_rewards.append(episode_reward)
    return episode_rewards

# 主函数：对比三种 epsilon 策略并统计 N_RUNS 结果

def main():
    strategies = [
        ("Exponential", epsilon_exponential, {'initial': 1.0, 'decay_rate': 0.99995}),
        ("Linear",      epsilon_linear,      {'initial': 1.0, 'min_epsilon': 0.05, 'max_episodes': EPISODES}),
        ("InverseTime", epsilon_inverse_time,{'initial': 1.0, 'k': 0.0005}),
    ]

    plt.figure(figsize=(12, 6))

    for name, func, kwargs in strategies:
        print(f"Running strategy: {name} for {N_RUNS} runs...")
        smoothed_runs = []

        # 多次运行并收集平滑曲线
        for run in range(N_RUNS):
            rewards = run_q_learning(func, kwargs)
            window = 100
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            smoothed_runs.append(smoothed)

        smoothed_runs = np.array(smoothed_runs)
        mean_curve = smoothed_runs.mean(axis=0)
        std_curve  = smoothed_runs.std(axis=0)
        episodes = np.arange(len(mean_curve))

        # 绘制均值曲线及 95% 置信区间阴影
        plt.plot(episodes, mean_curve, label=name)
        plt.fill_between(
            episodes,
            mean_curve - 1.96 * std_curve,
            mean_curve + 1.96 * std_curve,
            alpha=0.2
        )

    plt.xlabel("Episode (smoothed)")
    plt.ylabel("Average Reward")
    plt.title("Comparison of Epsilon Decay Strategies (\u00B12*1.96*std)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
