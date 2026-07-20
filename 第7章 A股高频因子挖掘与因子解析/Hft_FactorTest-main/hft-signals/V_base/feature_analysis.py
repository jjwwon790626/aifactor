import os
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from numba import jit, njit
from concurrent.futures import ProcessPoolExecutor
import struct
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf
from multiprocessing import Pool
from functools import partial
from collections import Counter
import math
import argparse

import sys
# 添加 hft 库路径
hft_signals_path = '/Users/wook/Downloads/FactorTest1/hft-signals'
if hft_signals_path not in sys.path:
    sys.path.insert(0, hft_signals_path)

# 修复后的数据读取函数
def read_binary_file_to_series(filename, start_idx, step):
    """
    读取二进制文件并直接返回一个 Polars Series。
    """
    try:
        with open(filename, 'rb') as f:
            data = f.read()

        # 解析二进制数据
        parsed_data = parse_binary_data(data)

        # 筛选数据：从 start_idx 开始，每隔 step 个数读取
        selected_data = parsed_data[start_idx::step]

        # 转换为 Polars Series
        if selected_data:
            return pl.Series(selected_data)
        else:
            return pl.Series([])

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File '{filename}' not found.")
    except Exception as e:
        raise Exception(f"Error: {e}")

def parse_binary_data(data):
    """
    解析二进制数据为double数组
    """
    try:
        double_size = 8
        num_doubles = len(data) // double_size
        doubles = struct.unpack(f'{num_doubles}d', data)
        return doubles
    except struct.error:
        return None

# 修复后的特征重构函数
def reshape_feature_data_correct(feature_data, num_features, num_timepoints):
    """
    正确重构列优先存储的特征数据
    
    参数:
    - feature_data: 一维特征数据数组
    - num_features: 特征数量
    - num_timepoints: 时间点数量
    
    返回:
    - reshaped_data: (时间点数, 特征数) 的二维数组
    """
    # 将一维数据重塑为 (特征数, 时间点数)，然后转置
    feature_matrix = np.array(feature_data).reshape(num_features, num_timepoints)
    # 转置得到 (时间点数, 特征数)
    return feature_matrix.T

# 其他函数保持不变...
def calculate_numeric_stats(feature, label):
    """使用Numba加速计算统计指标"""
    res = np.zeros(5)
    try:
        res[0] = np.corrcoef(feature, label)[0, 1]  # label_corr
        res[1] = pacf(feature, nlags=1)[1]  # lag_1_pacf
    except np.linalg.LinAlgError:
        print("Warning: Singular matrix encountered in PACF calculation. Setting PACF to NaN.")
        res[1] = np.nan
    except Exception as e:
        print(f"Unexpected error in PACF calculation: {e}")
        res[1] = np.nan

    res[2] = np.std(feature)  # std_var
    res[3] = np.mean((feature - np.mean(feature)) ** 3) / (np.std(feature) ** 3)  # skewness
    res[4] = np.mean((feature - np.mean(feature)) ** 4) / (np.std(feature) ** 4) - 3  # kurtosis
    return res

@njit
def calculate_pacf_numba(y, nlags):
    """使用Numba加速PACF计算"""
    n = len(y)
    pacf_values = np.zeros(nlags + 1)
    pacf_values[0] = 1.0

    for lag in range(1, nlags + 1):
        y_lagged = y[:-lag]
        y_current = y[lag:]
        pacf_values[lag] = np.corrcoef(y_current, y_lagged)[0, 1]

    return pacf_values

def calculate_and_plot_pacf(factor, feature_data, output_path):
    """计算PACF并生成图表"""
    y = feature_data
    pacf_values = calculate_pacf_numba(y, nlags=20)

    plt.figure()
    plt.bar(range(len(pacf_values)), pacf_values, alpha=0.7)
    plt.title(f"PACF for {factor}")
    plt.xlabel("Lag")
    plt.ylabel("PACF")
    plt.savefig(output_path)
    plt.close()

@njit
def calculate_residual_sum_of_squares(y, y_pred):
    """计算残差平方和"""
    residuals = y - y_pred
    rss = np.sum(residuals ** 2)
    return rss

@njit
def ols_regression(y, X):
    """使用最小二乘法拟合线性回归模型"""
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    
    XtX = np.dot(X.T, X)
    XtX_inv = np.linalg.inv(XtX)
    XtY = np.dot(X.T, y)
    beta = np.dot(XtX_inv, XtY)
    
    y_pred = np.dot(X, beta)
    rss = calculate_residual_sum_of_squares(y, y_pred)
    tss = np.sum((y - np.mean(y)) ** 2)
    ess = tss - rss
    
    n = X.shape[0]
    k = X.shape[1]
    df_model = k - 1
    df_residual = n - k
    
    mse_model = ess / df_model
    mse_residual = rss / df_residual
    F_value = mse_model / mse_residual
    
    se = np.sqrt(np.diag(XtX_inv) * mse_residual)
    t_values = beta / se
    
    return beta, rss, t_values, F_value

@njit
def calculate_group_means_sorted(feature, label, n_groups):
    """使用排序分组方式计算每组的均值 - 修正版本：基于feature分组，计算label均值"""
    random_indices = np.random.permutation(len(feature))
    shuffled_label = label[random_indices]
    shuffled_feature = feature[random_indices]

    # 修改：基于feature进行排序，而不是label
    sorted_indices = np.argsort(shuffled_feature)
    sorted_feature = shuffled_feature[sorted_indices]
    sorted_label = shuffled_label[sorted_indices]

    group_size = len(feature) // n_groups
    group_means = np.empty(n_groups, dtype=np.float64)

    for j in range(n_groups):
        if j == n_groups - 1:
            # 修改：计算label的均值，而不是feature的均值
            group_labels = sorted_label[j * group_size:]
        else:
            group_labels = sorted_label[j * group_size:(j + 1) * group_size]

        if len(group_labels) > 0:
            group_means[j] = np.mean(group_labels)
        else:
            group_means[j] = np.nan

    for j in range(1, n_groups):
        if np.isnan(group_means[j]):
            group_means[j] = group_means[j - 1]

    return group_means

def calculate_and_plot_partition_mean(factor, feature_data, label, partition_mean_path, group_num):
    """分组计算均值并生成图表 - 修正版本"""
    group_means = calculate_group_means_sorted(feature_data, label, group_num)

    plt.figure(figsize=(15, 8))
    x = range(1, len(group_means) + 1)
    plt.bar(x, group_means, color='skyblue', edgecolor='black')

    # 修改标题和轴标签以反映正确的含义
    plt.title(f"Group Means for {factor} (Based on Feature)")
    plt.xlabel("Feature Group")  # 横轴是特征的分组
    plt.ylabel("Mean Label Value")  # 纵轴是标签的均值
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(x)

    plt.savefig(partition_mean_path)
    plt.close()

@njit
def calculate_group_data_sorted(feature: np.ndarray, label: np.ndarray, n_groups: int):
    """使用排序分组方式获取每组的标签值数据"""
    random_indices = np.random.permutation(len(feature))
    shuffled_feature = feature[random_indices]
    shuffled_label = label[random_indices]

    sorted_indices = np.argsort(shuffled_feature)
    sorted_feature = shuffled_feature[sorted_indices]
    sorted_label = shuffled_label[sorted_indices]

    group_size = len(feature) // n_groups
    group_labels = []

    for j in range(n_groups):
        if j == n_groups - 1:
            group_labels.append(sorted_label[j * group_size:])
        else:
            group_labels.append(sorted_label[j * group_size:(j + 1) * group_size])

    return group_labels

def calculate_and_plot_boxplot(factor, feature_data, label, boxplot_path, group_num):
    """分组计算数据并生成箱线图"""
    feature_data = np.asarray(feature_data)
    label = np.asarray(label)

    group_data = calculate_group_data_sorted(feature_data, label, group_num)

    plt.figure(figsize=(15, 8))
    plt.boxplot(group_data, patch_artist=True, showmeans=False, meanline=False,
                boxprops=dict(facecolor='lightblue', color='black'),
                medianprops=dict(color='red'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                flierprops=dict(marker=''),
                showfliers=False)

    plt.title(f"Boxplot for {factor} (Grouped by Feature)", fontsize=16)
    plt.xlabel("Group", fontsize=14)
    plt.ylabel("Label Value", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(range(1, group_num + 1))

    os.makedirs(os.path.dirname(boxplot_path), exist_ok=True)
    plt.savefig(boxplot_path)
    plt.close()

# 修复后的 FeatureAnalyzer 类
class FeatureAnalyzer:
    def __init__(self, data: np.ndarray, label: np.ndarray, factornames: str, output_base_path: str):
        """
        初始化 FeatureAnalyzer 类 - 修改为接受 numpy 数组
        """
        sns.set_theme(style='white', font_scale=1.5)

        self.label = label
        self.feature_name = factornames
        self.feature_data = data

        # 限制数据范围
        self.feature_data = np.clip(self.feature_data, -1e8, 1e8)
        self.label = np.clip(self.label, -1e8, 1e8)

        # 去掉 NaN 和 Inf 值对应的位置
        valid_mask = (
            ~np.isnan(self.feature_data) & ~np.isnan(self.label) &
            ~np.isinf(self.feature_data) & ~np.isinf(self.label)
        )
        self.feature_data = self.feature_data[valid_mask]
        self.label = self.label[valid_mask]

        # 初始化输出路径
        self.output_paths = {
            'hist': os.path.join(output_base_path, 'feature_hist'),
            'hist_trimmed': os.path.join(output_base_path, 'trimmed_feature_hist'),
            'plots': os.path.join(output_base_path, 'featureplot'),
            'stats': os.path.join(output_base_path, 'featurestat'),
            'pacf': os.path.join(output_base_path, 'pacfplot'),
            'regression': os.path.join(output_base_path, 'labelregressionsummary'),
            'partition_mean': os.path.join(output_base_path, 'featurepartitionmean_10group'),
            'boxplot': os.path.join(output_base_path, 'trimmed_boxplot'),
            'binary_regression': os.path.join(output_base_path, 'binaryregression'),
        }

        for path in self.output_paths.values():
            os.makedirs(path, exist_ok=True)

    def generate_histogram_trimmed(self, bins=100):
        output_file = os.path.join(self.output_paths['hist_trimmed'], f"{self.feature_name}_trimmed_histogram.png")

        lower_bound = np.quantile(self.feature_data, 0.05)
        upper_bound = np.quantile(self.feature_data, 0.95)

        trimmed_data = self.feature_data[(self.feature_data >= lower_bound) & (self.feature_data <= upper_bound)]

        plt.figure(figsize=(10, 6))
        sns.histplot(trimmed_data, bins=bins, kde=True, color='green')
        plt.xlabel(self.feature_name)
        plt.ylabel("Frequency")
        plt.title(f"Trimmed Histogram of {self.feature_name} (5%-95%)")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.savefig(output_file)
        plt.close()

    def generate_histogram(self, bins=100):
        """生成单一特征的直方图"""
        output_file = os.path.join(self.output_paths['hist'], f"{self.feature_name}_histogram.png")

        plt.figure(figsize=(10, 6))
        sns.histplot(self.feature_data, bins=bins, kde=True, color='blue')
        plt.xlabel(self.feature_name)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {self.feature_name}")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.savefig(output_file)
        plt.close()

    def generate_scatter_plot(self):
        """生成单一特征的散点图"""
        output_file = os.path.join(self.output_paths['plots'], f"{self.feature_name}_scatter_plot.png")
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.feature_data, y=self.label)
        plt.xlabel(self.feature_name)
        plt.ylabel("Label")
        plt.title(f"Scatter Plot of {self.feature_name} vs Label")
        plt.savefig(output_file)
        plt.close()

    def calculate_statistics(self):
        """计算单一特征的统计指标"""
        stats = calculate_numeric_stats(self.feature_data, self.label)

        df = pd.DataFrame([stats])
        df.columns = ['label_corr', 'lag_1_pacf', 'std_var', 'skewness', 'kurtosis']
        output_file = os.path.join(self.output_paths['stats'], f"{self.feature_name}_stats.csv")
        df.to_csv(output_file, index=False)

    def generate_pacf_plot(self):
        """生成单一特征的PACF图"""
        output_file = os.path.join(self.output_paths['pacf'], f"{self.feature_name}_pacf_plot.png")
        calculate_and_plot_pacf(self.feature_name, self.feature_data, output_file)

    def perform_regression_analysis(self):
        """执行单一特征的回归分析"""
        y = self.label
        X = self.feature_data.reshape(-1, 1)
        
        beta, rss, t_values, F_value = ols_regression(y, X)
        
        output_file = os.path.join(self.output_paths['regression'], f"{self.feature_name}_summary.txt")
        with open(output_file, "w") as f:
            f.write(f"Coefficients: {beta}\n")
            f.write(f"Residual Sum of Squares: {rss}\n")
            f.write(f"t-values: {t_values}\n")
            f.write(f"F-value: {F_value}\n")

    def perform_binary_regression_analysis(self):
        y = (self.label > 0).astype(float)
        X = self.feature_data.reshape(-1, 1)
        
        beta, rss, t_values, F_value = ols_regression(y, X)
        
        output_file = os.path.join(self.output_paths['binary_regression'], f"{self.feature_name}_summary.txt")
        with open(output_file, "w") as f:
            f.write(f"Coefficients: {beta}\n")
            f.write(f"Residual Sum of Squares: {rss}\n")
            f.write(f"t-values: {t_values}\n")
            f.write(f"F-value: {F_value}\n")

    def feature_partition_mean_to_csv_and_plot(self, group_num=10):
        """计算分位点并生成分组均值柱状图"""
        output_file_plot = os.path.join(self.output_paths['partition_mean'], f"{self.feature_name}_partition_mean_plot.png")
        calculate_and_plot_partition_mean(self.feature_name, self.feature_data, self.label, output_file_plot, group_num)

    def feature_boxplot(self, group_num=10):
        """计算分位点并生成分组均值柱状图"""
        output_file_plot = os.path.join(self.output_paths['boxplot'], f"{self.feature_name}_boxplot.png")
        calculate_and_plot_boxplot(self.feature_name, self.feature_data, self.label, output_file_plot, group_num)

# 修复后的主处理函数
def process_feature(feature_base_path, label_file, feature_name_file, output_base_path):
    """
    处理单个特征的分析任务 - 修复版本
    """
    print("读取数据...")
    
    # 读取标签数据
    label_series = read_binary_file_to_series(label_file, 0, 1)
    label = label_series.to_numpy()  # 转换为 numpy 数组
    
    # 读取特征数据
    feature_series = read_binary_file_to_series(feature_base_path, 0, 1)
    feature_data = feature_series.to_numpy()  # 转换为 numpy 数组
    
    # 读取特征名称
    _feature_file = pd.read_csv(feature_name_file, header=None)
    _feature_list = list(_feature_file.iloc[:, 0].str[:-4])
    num_features = len(_feature_list)
    
    print(f"特征数量: {num_features}")
    print(f"总数据点数: {len(feature_data)}")
    print(f"标签数量: {len(label)}")
    
    # 计算时间点数量
    num_timepoints = len(feature_data) // num_features
    print(f"时间点数量: {num_timepoints}")
    
    # 验证数据完整性
    if len(feature_data) != num_features * num_timepoints:
        print(f"警告: 数据长度不匹配!")
        print(f"特征数据长度: {len(feature_data)}")
        print(f"期望长度: {num_features * num_timepoints}")
        return
    
    # 正确重构特征数据
    feature_matrix = reshape_feature_data_correct(feature_data, num_features, num_timepoints)
    print(f"重构后的特征矩阵形状: {feature_matrix.shape}")
    
    # 确保标签长度匹配
    if len(label) != num_timepoints:
        print(f"警告: 标签长度 {len(label)} 与时间点数量 {num_timepoints} 不匹配")
        min_length = min(len(label), num_timepoints)
        label = label[:min_length]
        feature_matrix = feature_matrix[:min_length, :]
        print(f"调整后的数据长度: {min_length}")
    
    # 逐个处理每个特征
    for i, feature_name in enumerate(_feature_list):
        print(f"处理特征: {feature_name} ({i+1}/{num_features})")
        
        # 提取单个特征的数据
        single_feature_data = feature_matrix[:, i]
        
        # 创建分析器
        analyzer = FeatureAnalyzer(single_feature_data, label, feature_name, output_base_path)

        # 执行分析任务
        try:
            analyzer.generate_scatter_plot()
            analyzer.calculate_statistics()
            analyzer.generate_pacf_plot()
            analyzer.perform_regression_analysis()
            analyzer.feature_partition_mean_to_csv_and_plot(group_num=10)
            analyzer.perform_binary_regression_analysis()
            analyzer.generate_histogram()
            analyzer.feature_boxplot(group_num=10)
            analyzer.generate_histogram_trimmed()
            
            print(f"特征 {feature_name} 分析完成")
        except Exception as e:
            print(f"处理特征 {feature_name} 时出错: {e}")
        
        del analyzer

def parse_regression_summary_file(file_path):
    """解析单个回归分析结果文件"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        feature_name = os.path.basename(file_path).replace('_summary.txt', '')
        
        data = {'feature_name': feature_name}
        
        # 解析 Coefficients
        coef_line = [line for line in content.split('\n') if line.startswith('Coefficients:')][0]
        coef_str = coef_line.split('Coefficients: ')[1].strip()
        
        coef_str = coef_str.strip('[]')
        coef_parts = coef_str.split()
        
        data['intercept'] = float(coef_parts[0])
        data['slope'] = float(coef_parts[1])
        
        # 解析其他指标
        rss_line = [line for line in content.split('\n') if line.startswith('Residual Sum of Squares:')][0]
        data['rss'] = float(rss_line.split('Residual Sum of Squares: ')[1])
        
        t_line = [line for line in content.split('\n') if line.startswith('t-values:')][0]
        t_str = t_line.split('t-values: ')[1].strip()
        
        t_str = t_str.strip('[]')
        t_parts = t_str.split()
        
        data['t_intercept'] = float(t_parts[0])
        data['t_slope'] = float(t_parts[1])
        
        f_line = [line for line in content.split('\n') if line.startswith('F-value:')][0]
        data['f_value'] = float(f_line.split('F-value: ')[1])
        
        return data
        
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return None

def collect_regression_results(regression_summary_path, output_path):
    """收集所有回归分析结果并整理成表格"""
    try:
        os.makedirs(output_path, exist_ok=True)
        
        txt_files = [f for f in os.listdir(regression_summary_path) if f.endswith('_summary.txt')]
        
        if not txt_files:
            print(f"No summary files found in {regression_summary_path}")
            return
        
        results = []
        for txt_file in txt_files:
            file_path = os.path.join(regression_summary_path, txt_file)
            parsed_data = parse_regression_summary_file(file_path)
            if parsed_data:
                results.append(parsed_data)
        
        if not results:
            print("No valid data parsed from files")
            return
        
        df = pd.DataFrame(results)
        
        column_order = ['feature_name', 'intercept', 'slope', 't_intercept', 't_slope', 'rss', 'f_value']
        df = df[column_order]
        
        df = df.sort_values('f_value', ascending=False)
        
        output_file = os.path.join(output_path, 'regression_summary_results.csv')
        df.to_csv(output_file, index=False)
        
        print(f"Regression results saved to: {output_file}")
        print(f"Total features analyzed: {len(df)}")
        
        return df
        
    except Exception as e:
        print(f"Error collecting regression results: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Feature analysis script.")
    parser.add_argument("--feature_base_path", type=str, required=False, 
                       default="/Users/wook/Downloads/FactorTest1/feature_test_data/feature/combined_features")
    parser.add_argument("--label_file", type=str, required=False, 
                       default="/Users/wook/Downloads/FactorTest1/feature_test_data/label/combined_labels")
    parser.add_argument("--feature_name", type=str, required=False, 
                       default="/Users/wook/Downloads/FactorTest1/feature_error_test/factorname.csv")
    parser.add_argument("--output_base_path", type=str, required=False, 
                       default="/Users/wook/Downloads/FactorTest1/feature_analysis")
    args = parser.parse_args()

    # 调用特征处理函数
    process_feature(args.feature_base_path, args.label_file, args.feature_name, args.output_base_path)
    
    # 收集和整理回归分析结果
    regression_summary_path = os.path.join(args.output_base_path, "labelregressionsummary")
    output_path = args.output_base_path
    
    print("\nCollecting regression analysis results...")
    collect_regression_results(regression_summary_path, output_path)

if __name__ == "__main__":
    main()