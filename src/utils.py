"""
工具函数模块
包含通用工具函数，如softmax，日志打印等
"""
import numpy as np
from src.config import VERBOSE

def log_print(message):
    """根据VERBOSE标志输出日志信息
    
    Args:
        message (str): 要打印的消息
    """
    if VERBOSE:
        print(message)

def softmax(x, tau=1.0):
    """softmax函数：将网络输出转换为概率分布
    
    Args:
        x (array-like): 输入向量
        tau (float, optional): 温度参数. Defaults to 1.0.
    
    Returns:
        np.ndarray: 概率分布
    """
    x = np.array(x) / tau
    e_x = np.exp(x - np.max(x))
    return e_x / (np.sum(e_x) + 1e-8)

def calculate_confidence_interval(data, confidence=0.95):
    """计算置信区间
    
    Args:
        data (array-like): 数据
        confidence (float, optional): 置信度. Defaults to 0.95.
    
    Returns:
        float: 置信区间的半宽
    """
    from scipy import stats
    if len(data) <= 1:
        return 0  # 如果只有一个数据点，返回0
    n = len(data)
    m = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return h
