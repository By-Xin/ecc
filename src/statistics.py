"""
统计分析模块
包含数据分析和结果比较的函数
"""
import numpy as np
from scipy import stats
from src.utils import calculate_confidence_interval

def perform_wilcoxon_test(meta_rewards, vanilla_rewards, alpha=0.05):
    """执行Wilcoxon rank-sum test
    
    Args:
        meta_rewards (list): Meta-NEAT的奖励列表
        vanilla_rewards (list): Vanilla NEAT的奖励列表
        alpha (float, optional): 显著性水平. Defaults to 0.05.
    
    Returns:
        tuple: (统计量, p值, 是否显著)
    """
    statistic, p_value = stats.ranksums(meta_rewards, vanilla_rewards)
    is_significant = p_value < alpha
    return statistic, p_value, is_significant

def compare_results(generation_settings, meta_results, vanilla_results, confidence=0.95):
    """比较Meta-NEAT和Vanilla NEAT的结果，输出统计分析
    
    Args:
        generation_settings (list): 代数设置列表
        meta_results (list): Meta-NEAT结果 [(代数, 奖励列表), ...]
        vanilla_results (list): Vanilla NEAT结果 [(代数, 奖励列表), ...]
        confidence (float, optional): 置信度. Defaults to 0.95.
    """
    print("\n=== Summary of Results ===")
    for gens, rewards in meta_results:
        ci = calculate_confidence_interval(rewards, confidence)
        print(f"Meta-NEAT (Generations = {gens}): Mean Reward = {np.mean(rewards):.2f} ± {ci:.2f} ({confidence*100:.0f}% CI), Std = {np.std(rewards):.2f}")
    
    for gens, rewards in vanilla_results:
        ci = calculate_confidence_interval(rewards, confidence)
        print(f"Vanilla NEAT (Generations = {gens}): Mean Reward = {np.mean(rewards):.2f} ± {ci:.2f} ({confidence*100:.0f}% CI), Std = {np.std(rewards):.2f}")
    
    print("\n=== Wilcoxon Rank-Sum Test Results ===")
    # 比较结论（基于Wilcoxon test和置信区间）
    for gens, m_rewards in meta_results:
        for g_v, v_rewards in vanilla_results:
            if gens == g_v:
                m_mean = np.mean(m_rewards)
                v_mean = np.mean(v_rewards)
                m_ci = calculate_confidence_interval(m_rewards, confidence)
                v_ci = calculate_confidence_interval(v_rewards, confidence)
                
                # 执行Wilcoxon test
                statistic, p_value, is_significant = perform_wilcoxon_test(m_rewards, v_rewards)
                
                print(f"\nAt {gens} generations:")
                print(f"Meta-NEAT: {m_mean:.2f} ± {m_ci:.2f}")
                print(f"Vanilla NEAT: {v_mean:.2f} ± {v_ci:.2f}")
                print(f"Wilcoxon test: statistic = {statistic:.4f}, p-value = {p_value:.4f}")
                
                if is_significant:
                    if m_mean > v_mean:
                        print("Meta-NEAT significantly outperforms Vanilla NEAT (p < 0.05)")
                    else:
                        print("Vanilla NEAT significantly outperforms Meta-NEAT (p < 0.05)")
                else:
                    print("No significant difference between Meta-NEAT and Vanilla NEAT (p >= 0.05)")
