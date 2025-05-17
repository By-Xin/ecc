"""
可视化模块
包含绘制性能比较图、训练过程图和局部搜索统计图的函数
"""
import numpy as np
import matplotlib.pyplot as plt
from src.utils import calculate_confidence_interval
from src.config import local_search_stats, MAX_LOCAL_TRIALS

def plot_performance_comparison(generation_settings, meta_data, vanilla_data, confidence=0.95):
    """绘制性能比较图函数
    
    Args:
        generation_settings (list): 代数设置列表
        meta_data (list): Meta-NEAT数据 [(代数, 奖励列表), ...]
        vanilla_data (list): Vanilla NEAT数据 [(代数, 奖励列表), ...]
        confidence (float, optional): 置信度. Defaults to 0.95.
    """
    plt.figure(figsize=(16, 10))
    
    # 提取每代的平均值和置信区间
    x = generation_settings
    
    # Meta-NEAT数据
    y_meta = [np.mean(rewards) for _, rewards in meta_data]
    ci_meta = [calculate_confidence_interval(rewards, confidence) for _, rewards in meta_data]
    std_meta = [np.std(rewards) for _, rewards in meta_data]
    
    # Vanilla NEAT数据
    y_vanilla = [np.mean(rewards) for _, rewards in vanilla_data]
    ci_vanilla = [calculate_confidence_interval(rewards, confidence) for _, rewards in vanilla_data]
    std_vanilla = [np.std(rewards) for _, rewards in vanilla_data]
    
    # 绘制带置信区间的填充区域
    plt.plot(x, y_meta, '-o', markersize=8, label='Meta-NEAT', color='blue', linewidth=2.5)
    plt.fill_between(x, 
                     np.array(y_meta) - np.array(ci_meta), 
                     np.array(y_meta) + np.array(ci_meta), 
                     alpha=0.2, color='blue')
    
    plt.plot(x, y_vanilla, '-s', markersize=8, label='Vanilla NEAT', color='red', linewidth=2.5)
    plt.fill_between(x, 
                     np.array(y_vanilla) - np.array(ci_vanilla), 
                     np.array(y_vanilla) + np.array(ci_vanilla), 
                     alpha=0.2, color='red')
    
    plt.xlabel('进化代数', fontsize=16)
    plt.ylabel('平均奖励', fontsize=16)
    plt.title(f'Meta-NEAT vs Vanilla NEAT 性能对比 ({confidence*100:.0f}%置信区间)', fontsize=18)
    
    # 设置x轴刻度和网格更细致
    plt.xticks(generation_settings)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # 保存图像
    plt.savefig('neat_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_progression(training_data_meta, training_data_vanilla, confidence=0.95):
    """绘制训练过程性能比较图函数
    
    Args:
        training_data_meta (list): Meta-NEAT训练数据，每个元素是一次运行的每代最佳适应度
        training_data_vanilla (list): Vanilla NEAT训练数据，每个元素是一次运行的每代最佳适应度
        confidence (float, optional): 置信度. Defaults to 0.95.
    """
    plt.figure(figsize=(16, 10))
    
    # 确定最大代数（基于数据长度）
    max_gens = max([len(data) for data in training_data_meta + training_data_vanilla])
    x = list(range(1, max_gens + 1))
    
    # 根据具体代数设置调整横轴宽度
    if max_gens <= 20:
        x_step = 1
    elif max_gens <= 40:
        x_step = 2
    else:
        x_step = 5
    
    # 计算每一代的平均值和置信区间
    y_meta = []
    ci_meta = []
    y_vanilla = []
    ci_vanilla = []
    
    # 对于每一代计算统计量
    for gen in range(max_gens):
        # 收集该代的所有数据点
        gen_data_meta = [run_data[gen] if gen < len(run_data) else None for run_data in training_data_meta]
        gen_data_meta = [d for d in gen_data_meta if d is not None]
        
        gen_data_vanilla = [run_data[gen] if gen < len(run_data) else None for run_data in training_data_vanilla]
        gen_data_vanilla = [d for d in gen_data_vanilla if d is not None]
        
        # 计算平均值和置信区间
        y_meta.append(np.mean(gen_data_meta) if gen_data_meta else 0)
        ci_meta.append(calculate_confidence_interval(gen_data_meta, confidence) if len(gen_data_meta) > 1 else 0)
        
        y_vanilla.append(np.mean(gen_data_vanilla) if gen_data_vanilla else 0)
        ci_vanilla.append(calculate_confidence_interval(gen_data_vanilla, confidence) if len(gen_data_vanilla) > 1 else 0)
    
    # 绘制带置信区间的线图
    plt.plot(x, y_meta, '-o', label='Meta-NEAT', color='blue', linewidth=2)
    plt.fill_between(x, np.array(y_meta) - np.array(ci_meta), np.array(y_meta) + np.array(ci_meta), 
                     color='blue', alpha=0.2)
    
    plt.plot(x, y_vanilla, '-s', label='Vanilla NEAT', color='red', linewidth=2)
    plt.fill_between(x, np.array(y_vanilla) - np.array(ci_vanilla), np.array(y_vanilla) + np.array(ci_vanilla), 
                     color='red', alpha=0.2)
    
    plt.xlabel('进化代数', fontsize=16)
    plt.ylabel('最佳适应度（奖励）', fontsize=16)
    plt.title(f'Meta-NEAT vs Vanilla NEAT 训练过程 ({confidence*100:.0f}%置信区间)', fontsize=18)
    
    # 根据训练代数设置合适的刻度密度
    plt.xticks(range(0, max_gens + 1, x_step))
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # 保存图像
    plt.savefig('neat_training_progression.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_local_search_statistics():
    """绘制局部搜索统计信息图函数"""
    from src.config import collect_statistics
    
    if not collect_statistics or len(local_search_stats['trials']) == 0:
        print("没有收集到局部搜索统计信息")
        return
    
    plt.figure(figsize=(16, 15))
    
    # 绘制局部搜索次数分布
    plt.subplot(3, 1, 1)
    plt.hist(local_search_stats['trials'], bins=range(1, MAX_LOCAL_TRIALS+2), alpha=0.7, 
             color='blue', edgecolor='black')
    plt.xlabel('局部搜索次数', fontsize=14)
    plt.ylabel('频率', fontsize=14)
    plt.title('局部搜索次数分布 (泊松分布)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 绘制改进次数分布
    plt.subplot(3, 1, 2)
    max_improvements = max(local_search_stats['improvements']) + 1
    plt.hist(local_search_stats['improvements'], bins=range(max_improvements+1), alpha=0.7,
             color='green', edgecolor='black')
    plt.xlabel('有效改进次数', fontsize=14)
    plt.ylabel('频率', fontsize=14)
    plt.title('局部搜索有效改进次数分布', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 绘制改进率随时间变化
    plt.subplot(3, 1, 3)
    window_size = min(50, len(local_search_stats['improvement_ratios']))
    if window_size > 0:
        moving_avg = []
        for i in range(len(local_search_stats['improvement_ratios']) - window_size + 1):
            moving_avg.append(np.mean(local_search_stats['improvement_ratios'][i:i+window_size]))
        
        plt.plot(range(window_size-1, len(local_search_stats['improvement_ratios'])), 
                moving_avg, color='red', linewidth=2)
        plt.xlabel('局部搜索索引', fontsize=14)
        plt.ylabel('改进率 (滑动平均)', fontsize=14)
        plt.title(f'局部搜索改进率变化 (窗口大小={window_size})', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('local_search_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计摘要
    trials_mean = np.mean(local_search_stats['trials'])
    trials_std = np.std(local_search_stats['trials'])
    improvements_mean = np.mean(local_search_stats['improvements'])
    improvement_ratio_mean = np.mean(local_search_stats['improvement_ratios'])
    
    print(f"\n=== 局部搜索统计摘要 ===")
    print(f"总局部搜索次数: {len(local_search_stats['trials'])}")
    print(f"平均每次尝试的搜索次数: {trials_mean:.2f} ± {trials_std:.2f}")
    print(f"平均每次改进次数: {improvements_mean:.2f}")
    print(f"平均改进率: {improvement_ratio_mean:.2f}")
