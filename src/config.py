"""
全局参数配置模块
在保留原有 Meta-NEAT / Vanilla-NEAT 参数的基础上，
追加 Hybrid (NEAT + DQN) 相关超参数。
"""

import numpy as np

# ===== 既有参数（保持不变） =====
GAME            = 'CartPole-v1'
CONFIG_PATH     = "./config/neat_config.txt"
EP_STEP         = 300                # 每个 episode 最大步数
NUM_GENERATIONS = 10                 # 进化代数
BASE_LOCAL_TRIALS = 5                # Meta-NEAT 内循环基准次数
MAX_LOCAL_TRIALS  = 15
NUM_RUNS        = 30
CONFIDENCE      = 0.95

USE_SOFTMAX = True
TAU         = 1.0                    # Soft-max 温度
C_UCB       = 0.5                    # （当前未使用）

SEED_BASE   = 42
VERBOSE     = False

# —— 局部搜索统计 —— #
collect_statistics = True
local_search_stats = {
    'trials': [],
    'improvements': [],
    'improvement_ratios': []
}

# =========================================================
# ===========  Hybrid (NEAT + DQN) 新增参数  ===============
# =========================================================
# 每一步使用 DQN 的初始概率 P
MIX_PROB   = 0.30
# 若希望随世代逐步降低 P，可在 neat_algorithms.py 内
# 乘以该衰减系数（不影响原有算法）
PROB_DECAY = 0.995

# ---------- DQN 超参数 ----------
RL_HIDDEN    = 64
RL_LR        = 1e-3
RL_GAMMA     = 0.99
RL_EPS_START = 1.0
RL_EPS_END   = 0.05
RL_EPS_DECAY = 5000        # 探索率指数衰减步数
BUFFER_SIZE  = 50_000
BATCH_SIZE   = 64

