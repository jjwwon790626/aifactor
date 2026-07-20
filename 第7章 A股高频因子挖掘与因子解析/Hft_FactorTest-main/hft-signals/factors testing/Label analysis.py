import struct
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def read_binary_label_file(filename):
    """
    读取二进制标签文件并返回numpy数组
    """
    try:
        with open(filename, 'rb') as f:
            data = f.read()
        
        # 解析二进制数据为double数组
        double_size = 8
        num_doubles = len(data) // double_size
        
        if len(data) % double_size != 0:
            print(f"警告: 文件大小不是8字节的整数倍，可能存在数据问题")
            print(f"文件大小: {len(data)} 字节")
            print(f"完整double数量: {num_doubles}")
            print(f"剩余字节: {len(data) % double_size}")
        
        # 解包为double数组
        doubles = struct.unpack(f'{num_doubles}d', data[:num_doubles * double_size])
        
        return np.array(doubles)
        
    except FileNotFoundError:
        raise FileNotFoundError(f"错误: 文件 '{filename}' 未找到")
    except Exception as e:
        raise Exception(f"读取文件时出错: {e}")

def analyze_label_distribution(labels, save_plots=True, output_dir="./label_analysis"):
    """
    全面分析标签数据的分布
    """
    import os
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("标签数据分布分析")
    print("=" * 60)
    
    # 1. 基本统计信息
    print("1. 基本统计信息:")
    print(f"   总样本数: {len(labels):,}")
    print(f"   最小值: {np.min(labels):.8f}")
    print(f"   最大值: {np.max(labels):.8f}")
    print(f"   均值: {np.mean(labels):.8f}")
    print(f"   中位数: {np.median(labels):.8f}")
    print(f"   标准差: {np.std(labels):.8f}")
    print(f"   方差: {np.var(labels):.8f}")
    
    # 2. 正负值分布
    print("\n2. 正负值分布:")
    positive_count = np.sum(labels > 0)
    negative_count = np.sum(labels < 0)
    zero_count = np.sum(labels == 0)
    
    positive_ratio = positive_count / len(labels)
    negative_ratio = negative_count / len(labels)
    zero_ratio = zero_count / len(labels)
    
    print(f"   正值数量: {positive_count:,} ({positive_ratio*100:.2f}%)")
    print(f"   负值数量: {negative_count:,} ({negative_ratio*100:.2f}%)")
    print(f"   零值数量: {zero_count:,} ({zero_ratio*100:.2f}%)")
    
    # 3. 分位数分析
    print("\n3. 分位数分析:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(labels, p)
        print(f"   {p:2d}%分位数: {value:.8f}")
    
    # 4. 异常值检测
    print("\n4. 异常值检测:")
    q1, q3 = np.percentile(labels, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers_lower = np.sum(labels < lower_bound)
    outliers_upper = np.sum(labels > upper_bound)
    total_outliers = outliers_lower + outliers_upper
    
    print(f"   IQR: {iqr:.8f}")
    print(f"   下界: {lower_bound:.8f}")
    print(f"   上界: {upper_bound:.8f}")
    print(f"   下端异常值: {outliers_lower:,}")
    print(f"   上端异常值: {outliers_upper:,}")
    print(f"   总异常值: {total_outliers:,} ({total_outliers/len(labels)*100:.2f}%)")
    
    # 5. 偏度和峰度
    print("\n5. 分布形状:")
    skewness = np.mean(((labels - np.mean(labels)) / np.std(labels)) ** 3)
    kurtosis = np.mean(((labels - np.mean(labels)) / np.std(labels)) ** 4) - 3
    
    print(f"   偏度 (Skewness): {skewness:.6f}")
    if skewness > 0.5:
        print("     -> 右偏分布 (正偏)")
    elif skewness < -0.5:
        print("     -> 左偏分布 (负偏)")
    else:
        print("     -> 接近对称分布")
    
    print(f"   峰度 (Kurtosis): {kurtosis:.6f}")
    if kurtosis > 0:
        print("     -> 尖峰分布")
    else:
        print("     -> 平峰分布")
    
    # 6. 生成可视化图表
    if save_plots:
        print(f"\n6. 生成可视化图表 (保存到 {output_dir})...")
        
        # 设置绘图风格
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        # 子图1: 直方图
        plt.subplot(2, 3, 1)
        plt.hist(labels, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('标签数据直方图', fontsize=14)
        plt.xlabel('标签值')
        plt.ylabel('频次')
        plt.grid(True, alpha=0.3)
        
        # 子图2: 箱线图
        plt.subplot(2, 3, 2)
        plt.boxplot(labels, patch_artist=True, 
                   boxprops=dict(facecolor='lightgreen'),
                   medianprops=dict(color='red', linewidth=2))
        plt.title('标签数据箱线图', fontsize=14)
        plt.ylabel('标签值')
        plt.grid(True, alpha=0.3)
        
        # 子图3: QQ图 (正态性检验)
        from scipy import stats
        plt.subplot(2, 3, 3)
        stats.probplot(labels, dist="norm", plot=plt)
        plt.title('Q-Q图 (正态性检验)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 子图4: 密度图
        plt.subplot(2, 3, 4)
        plt.hist(labels, bins=100, density=True, alpha=0.7, color='orange')
        plt.title('标签数据密度分布', fontsize=14)
        plt.xlabel('标签值')
        plt.ylabel('密度')
        plt.grid(True, alpha=0.3)
        
        # 子图5: 累积分布
        plt.subplot(2, 3, 5)
        sorted_labels = np.sort(labels)
        cumulative = np.arange(1, len(sorted_labels) + 1) / len(sorted_labels)
        plt.plot(sorted_labels, cumulative, linewidth=2)
        plt.title('累积分布函数', fontsize=14)
        plt.xlabel('标签值')
        plt.ylabel('累积概率')
        plt.grid(True, alpha=0.3)
        
        # 子图6: 时间序列图 (如果适用)
        plt.subplot(2, 3, 6)
        # 显示前1000个点的时间序列
        sample_size = min(1000, len(labels))
        plt.plot(range(sample_size), labels[:sample_size], alpha=0.7)
        plt.title(f'标签值时间序列 (前{sample_size}个点)', fontsize=14)
        plt.xlabel('索引')
        plt.ylabel('标签值')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/label_distribution_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存统计摘要到CSV
        summary_stats = {
            'Statistic': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', 
                         'Positive_Count', 'Negative_Count', 'Zero_Count',
                         'Positive_Ratio', 'Negative_Ratio', 'Skewness', 'Kurtosis'],
            'Value': [len(labels), np.mean(labels), np.std(labels), np.min(labels),
                     np.percentile(labels, 25), np.median(labels), np.percentile(labels, 75), np.max(labels),
                     positive_count, negative_count, zero_count,
                     positive_ratio, negative_ratio, skewness, kurtosis]
        }
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(f"{output_dir}/label_summary_statistics.csv", index=False)
        print(f"   统计摘要已保存到: {output_dir}/label_summary_statistics.csv")
    
    # 7. 诊断结论
    print("\n7. 诊断结论:")
    if negative_ratio > 0.8:
        print("   ⚠️  严重负偏: 超过80%的标签值为负")
        print("      -> 这解释了为什么特征分组图全是负值")
        print("      -> 可能的原因: 数据期间为下跌趋势、标签定义问题")
    elif negative_ratio > 0.6:
        print("   ⚠️  明显负偏: 超过60%的标签值为负")
        print("      -> 这可能导致大部分分组均值为负")
    elif abs(np.mean(labels)) < np.std(labels) * 0.1:
        print("   ✓  数据看起来相对平衡")
    
    if abs(skewness) > 2:
        print("   ⚠️  严重偏斜分布")
    elif abs(skewness) > 1:
        print("   ⚠️  中度偏斜分布")
    
    if total_outliers / len(labels) > 0.1:
        print("   ⚠️  异常值较多，可能需要清理")
    
    return {
        'positive_ratio': positive_ratio,
        'negative_ratio': negative_ratio,
        'mean': np.mean(labels),
        'std': np.std(labels),
        'skewness': skewness,
        'kurtosis': kurtosis,
        'outliers_ratio': total_outliers / len(labels)
    }

# 主函数
def main():
    label_file = "/Users/wook/Downloads/FactorTest1/feature_test_data/label/combined_labels"
    
    print("读取标签文件...")
    try:
        labels = read_binary_label_file(label_file)
        print(f"成功读取 {len(labels):,} 个标签值")
        
        # 分析分布
        stats = analyze_label_distribution(labels, save_plots=True, output_dir="./label_analysis")
        
        print("\n" + "=" * 60)
        print("分析完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()