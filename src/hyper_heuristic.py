import random
from src.config import MIX_PROB

def choose_policy() -> str:
    """伯努利采样；返回 'RL' 或 'NEAT'。"""
    return 'RL' if random.random() < MIX_PROB else 'NEAT'

