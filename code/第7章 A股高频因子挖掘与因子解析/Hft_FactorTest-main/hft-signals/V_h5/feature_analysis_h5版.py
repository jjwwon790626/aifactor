import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit
from statsmodels.tsa.stattools import pacf
import argparse
from tqdm import tqdm
import math

# H5数据读取和预处理函数
def load_and_filter_h5_data(feature_h5_path, label_h5_path, factor_name_path=None):
    """从H5文件读取特征和标签数据，过滤掉标签为0的行"""
    # 读取数据
    features_df = pd.read_hdf(feature_h5_path, key='features')
    labels_df = pd.read_hdf(label_h5_path, key='labels')
    labels = labels_df['labels'].values
    
    # 获取特征名称
    feature_names = features_df.columns.tolist()
    
    # 读取因子名称（如果提供）
    if factor_name_path and os.path.exists(factor_name_path):
        try:
            factor_names_df = pd.read_csv(factor_name_path, header=None)
            factor_names_from_file = factor_names_df[0].str.replace('.csv', '').tolist()
            if len(factor_names_from_file) == len(feature_names):
                feature_names = factor_names_from_file
        except:
            pass
    
    # 验证数据一致性
    if len(features_df) != len(labels):
        min_len = min(len(features_df), len(labels))
        features_df = features_df.iloc[:min_len]
        labels = labels[:min_len]
    
    # 过滤掉标签为0的行
    non_zero_mask = (labels != 0)
    features_df_filtered = features_df[non_zero_mask].copy()
    labels_filtered = labels[non_zero_mask].copy()
    
    # 转换数据类型
    for col in features_df_filtered.columns:
        features_df_filtered[col] = pd.to_numeric(features_df_filtered[col], errors='coerce')
    
    feature_matrix = features_df_filtered.astype(np.float64).values
    labels_filtered = labels_filtered.astype(np.float64)
    
    print(f"加载完成: {feature_matrix.shape[0]} 行, {len(feature_names)} 个特征 (已过滤标签为0的行)")
    
    return feature_matrix, labels_filtered, feature_names

# 统计函数
def calculate_numeric_stats(feature, label):
    """计算扩展的统计指标"""
    res = np.zeros(5)
    try:
        res[0] = np.corrcoef(feature, label)[0, 1]  # label_corr
        res[1] = pacf(feature, nlags=1)[1]  # lag_1_pacf
    except:
        res[1] = np.nan
    res[2] = np.std(feature)  # std_var
    res[3] = np.mean((feature - np.mean(feature)) ** 3) / (np.std(feature) ** 3)  # skewness
    res[4] = np.mean((feature - np.mean(feature)) ** 4) / (np.std(feature) ** 4) - 3  # kurtosis
    return res

@njit
def calculate_pacf_numba(y, nlags):
    """
    使用Numba加速PACF计算
    :param y: 时间序列数据
    :param nlags: 滞后阶数
    :return: PACF 值数组
    """
    n = len(y)
    pacf_values = np.zeros(nlags + 1)
    pacf_values[0] = 1.0  # 自相关系数为1

    for lag in range(1, nlags + 1):
        if lag >= n:
            pacf_values[lag] = 0.0
            continue
        y_lagged = y[:-lag]
        y_current = y[lag:]
        if len(y_current) > 0 and len(y_lagged) > 0:
            try:
                corr_matrix = np.corrcoef(y_current, y_lagged)
                if corr_matrix.shape == (2, 2):
                    pacf_values[lag] = corr_matrix[0, 1]
                else:
                    pacf_values[lag] = 0.0
            except:
                pacf_values[lag] = 0.0
        else:
            pacf_values[lag] = 0.0

    return pacf_values

@njit
def ols_regression(y, X):
    """使用最小二乘法拟合线性回归模型"""
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    XtX = np.dot(X.T, X)
    XtX_inv = np.linalg.inv(XtX)
    XtY = np.dot(X.T, y)
    beta = np.dot(XtX_inv, XtY)
    
    y_pred = np.dot(X, beta)
    rss = np.sum((y - y_pred) ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)
    ess = tss - rss
    
    n, k = X.shape
    mse_model = ess / (k - 1)
    mse_residual = rss / (n - k)
    F_value = mse_model / mse_residual
    
    se = np.sqrt(np.diag(XtX_inv) * mse_residual)
    t_values = beta / se
    
    return beta, rss, t_values, F_value

@njit
def calculate_group_means_sorted(feature, label, n_groups):
    """使用排序分组方式计算每组的均值"""
    random_indices = np.random.permutation(len(feature))
    shuffled_label = label[random_indices]
    shuffled_feature = feature[random_indices]
    
    sorted_indices = np.argsort(shuffled_feature)
    sorted_label = shuffled_label[sorted_indices]
    
    group_size = len(feature) // n_groups
    group_means = np.empty(n_groups, dtype=np.float64)
    
    for j in range(n_groups):
        if j == n_groups - 1:
            group_labels = sorted_label[j * group_size:]
        else:
            group_labels = sorted_label[j * group_size:(j + 1) * group_size]
        group_means[j] = np.mean(group_labels) if len(group_labels) > 0 else np.nan
    
    for j in range(1, n_groups):
        if np.isnan(group_means[j]):
            group_means[j] = group_means[j - 1]
    
    return group_means

@njit  
def calculate_group_data_sorted(feature: np.ndarray, label: np.ndarray, n_groups: int):
    """
    使用排序分组方式获取每组的标签值数据。
    :param feature: 单个特征的值 (NumPy 数组)
    :param label: 标签值 (NumPy 数组)
    :param n_groups: 分组数量
    :return: 每组的标签值列表
    """
    # 先打乱 feature 和 label
    random_indices = np.random.permutation(len(feature))
    shuffled_feature = feature[random_indices]
    shuffled_label = label[random_indices]

    # 根据 feature 排序
    sorted_indices = np.argsort(shuffled_feature)
    sorted_feature = shuffled_feature[sorted_indices]
    sorted_label = shuffled_label[sorted_indices]

    # 确定每组的边界索引
    group_size = len(feature) // n_groups
    group_labels = []

    for j in range(n_groups):
        if j == n_groups - 1:  # 最后一组
            group_data = sorted_label[j * group_size:]
        else:  # 其他组
            group_data = sorted_label[j * group_size:(j + 1) * group_size]
        group_labels.append(group_data)

    return group_labels

# 图表生成函数
def plot_partition_mean(factor, feature_data, label, output_path, group_num):
    """分组均值柱状图"""
    group_means = calculate_group_means_sorted(feature_data, label, group_num)
    plt.figure(figsize=(15, 8))
    plt.bar(range(1, len(group_means) + 1), group_means, color='skyblue', edgecolor='black')
    plt.title(f"Group Means for {factor}")
    plt.xlabel("Feature Group")
    plt.ylabel("Mean Label Value")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(output_path)
    plt.close()

def plot_histogram(feature_data, feature_name, output_path, bins=100):
    """生成直方图"""
    plt.figure(figsize=(10, 6))
    sns.histplot(feature_data, bins=bins, kde=True, color='blue')
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {feature_name}")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_path)
    plt.close()

def plot_histogram_trimmed(feature_data, feature_name, output_path, bins=100):
    """生成截断直方图(5%-95%)"""
    lower_bound = np.quantile(feature_data, 0.05)
    upper_bound = np.quantile(feature_data, 0.95)
    trimmed_data = feature_data[(feature_data >= lower_bound) & (feature_data <= upper_bound)]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(trimmed_data, bins=bins, kde=True, color='green')
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    plt.title(f"Trimmed Histogram of {feature_name} (5%-95%)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_path)
    plt.close()

def plot_scatter(feature_data, label, feature_name, output_path):
    """生成散点图"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=feature_data, y=label, s=60, alpha=0.8, color='cornflowerblue')
    plt.xlabel(feature_name)
    plt.ylabel("Label")
    plt.title(f"Scatter Plot of {feature_name} vs Label")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_pacf(factor, feature_data, output_path):
    """计算PACF并生成图表"""
    try:
        # 使用Numba加速计算PACF
        pacf_values = calculate_pacf_numba(feature_data, nlags=20)
        
        # 创建 PACF 图
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(pacf_values)), pacf_values, alpha=0.7, color='orange')
        plt.title(f"PACF for {factor}")
        plt.xlabel("Lag")
        plt.ylabel("PACF")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(output_path)
        plt.close()
    except Exception as e:
        print(f"PACF plot error for {factor}: {e}")

def plot_boxplot(factor, feature_data, label, output_path, group_num):
    """生成箱线图"""
    try:
        # 获取分组数据
        group_data = calculate_group_data_sorted(feature_data, label, group_num)
        
        # 转换为列表格式用于boxplot
        group_data_list = [group.tolist() if hasattr(group, 'tolist') else list(group) for group in group_data]
        
        plt.figure(figsize=(15, 8))
        plt.boxplot(group_data_list, patch_artist=True, showmeans=False, meanline=False,
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
        plt.savefig(output_path)
        plt.close()
    except Exception as e:
        print(f"Boxplot error for {factor}: {e}")

# 特征分析器类
class FeatureAnalyzer:
    def __init__(self, data: np.ndarray, label: np.ndarray, feature_name: str, output_base_path: str):
        sns.set_theme(style='white', font_scale=1.5)
        
        self.feature_name = feature_name
        
        # 确保数据类型正确
        try:
            self.feature_data = np.asarray(data, dtype=np.float64)
            self.label = np.asarray(label, dtype=np.float64)
        except:
            self.feature_data = np.array([])
            self.label = np.array([])
            return
        
        if len(self.label) == 0 or len(self.feature_data) == 0:
            return
        
        # 数据清洗
        self.feature_data = np.clip(self.feature_data, -1e8, 1e8)
        self.label = np.clip(self.label, -1e8, 1e8)
        
        # 移除无效值
        try:
            feature_valid = ~(pd.isna(self.feature_data) | np.isinf(self.feature_data))
            label_valid = ~(pd.isna(self.label) | np.isinf(self.label))
            valid_mask = feature_valid & label_valid
            self.feature_data = self.feature_data[valid_mask]
            self.label = self.label[valid_mask]
        except:
            finite_mask = np.isfinite(self.feature_data) & np.isfinite(self.label)
            self.feature_data = self.feature_data[finite_mask]
            self.label = self.label[finite_mask]
        
        # 输出路径设置 - 添加新的路径
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
    
    def has_data(self):
        return len(self.label) > 0 and len(self.feature_data) > 0
    
    def analyze(self):
        """执行完整的特征分析"""
        if not self.has_data():
            return
            
        try:
            # 生成直方图
            plot_histogram(self.feature_data, self.feature_name, 
                          os.path.join(self.output_paths['hist'], f"{self.feature_name}_histogram.png"))
            
            # 生成截断直方图
            plot_histogram_trimmed(self.feature_data, self.feature_name,
                                 os.path.join(self.output_paths['hist_trimmed'], f"{self.feature_name}_trimmed_histogram.png"))
            
            # 生成散点图
            plot_scatter(self.feature_data, self.label, self.feature_name,
                        os.path.join(self.output_paths['plots'], f"{self.feature_name}_scatter_plot.png"))
            
            # 生成PACF图
            plot_pacf(self.feature_name, self.feature_data,
                     os.path.join(self.output_paths['pacf'], f"{self.feature_name}_pacf_plot.png"))
            
            # 生成分组均值图
            if len(self.feature_data) >= 10:
                plot_partition_mean(self.feature_name, self.feature_data, self.label,
                                   os.path.join(self.output_paths['partition_mean'], f"{self.feature_name}_partition_mean_plot.png"), 10)
            
            # 生成箱线图
            if len(self.feature_data) >= 10:
                plot_boxplot(self.feature_name, self.feature_data, self.label,
                           os.path.join(self.output_paths['boxplot'], f"{self.feature_name}_boxplot.png"), 10)
            
            # 统计分析
            stats = calculate_numeric_stats(self.feature_data, self.label)
            df = pd.DataFrame([stats], columns=['label_corr', 'lag_1_pacf', 'std_var', 'skewness', 'kurtosis'])
            df.to_csv(os.path.join(self.output_paths['stats'], f"{self.feature_name}_stats.csv"), index=False)
            
            # 线性回归分析
            y = self.label
            X = self.feature_data.reshape(-1, 1)
            beta, rss, t_values, F_value = ols_regression(y, X)
            
            with open(os.path.join(self.output_paths['regression'], f"{self.feature_name}_summary.txt"), "w") as f:
                f.write(f"Sample size: {len(y)}\n")
                f.write(f"Coefficients: {beta}\n")
                f.write(f"Residual Sum of Squares: {rss}\n")
                f.write(f"t-values: {t_values}\n")
                f.write(f"F-value: {F_value}\n")
            
            # 二元回归分析
            y_binary = (self.label > 0).astype(float)
            beta_binary, rss_binary, t_values_binary, F_value_binary = ols_regression(y_binary, X)
            
            with open(os.path.join(self.output_paths['binary_regression'], f"{self.feature_name}_summary.txt"), "w") as f:
                f.write(f"Sample size: {len(y_binary)}\n")
                f.write(f"Coefficients: {beta_binary}\n")
                f.write(f"Residual Sum of Squares: {rss_binary}\n")
                f.write(f"t-values: {t_values_binary}\n")
                f.write(f"F-value: {F_value_binary}\n")
                
        except Exception as e:
            print(f"Analysis error for {self.feature_name}: {e}")

# 主处理函数
def process_features_from_h5(feature_h5_path, label_h5_path, output_base_path, factor_name_path=None):
    """从H5文件处理特征：先加载数据并过滤标签为0的行，然后进行特征分析"""
    # 加载和过滤数据
    feature_matrix, labels, feature_names = load_and_filter_h5_data(
        feature_h5_path, label_h5_path, factor_name_path
    )
    
    if len(labels) == 0:
        print("❌ 过滤后没有剩余数据")
        return
    
    # 分析特征
    successful_count = 0
    for i, feature_name in enumerate(tqdm(feature_names, desc="分析特征")):
        try:
            single_feature_data = feature_matrix[:, i]
            analyzer = FeatureAnalyzer(single_feature_data, labels, feature_name, output_base_path)
            
            if analyzer.has_data():
                analyzer.analyze()
                successful_count += 1
                
        except Exception as e:
            tqdm.write(f"❌ {feature_name}: {e}")
        finally:
            if 'analyzer' in locals():
                del analyzer
    
    print(f"✅ 完成! 成功分析 {successful_count}/{len(feature_names)} 个特征")

# 结果收集函数
def collect_results(regression_path, output_path):
    """收集回归分析结果"""
    txt_files = [f for f in os.listdir(regression_path) if f.endswith('_summary.txt')]
    results = []
    
    for txt_file in txt_files:
        try:
            with open(os.path.join(regression_path, txt_file), 'r') as f:
                content = f.read()
            
            feature_name = os.path.basename(txt_file).replace('_summary.txt', '')
            data = {'feature_name': feature_name}
            
            for line in content.split('\n'):
                if line.startswith('Sample size:'):
                    data['sample_size'] = int(line.split(': ')[1])
                elif line.startswith('Coefficients:'):
                    coef_str = line.split(': ')[1].strip('[]')
                    coef_parts = coef_str.split()
                    data['intercept'] = float(coef_parts[0])
                    data['slope'] = float(coef_parts[1])
                elif line.startswith('t-values:'):
                    t_str = line.split(': ')[1].strip('[]')
                    t_parts = t_str.split()
                    data['t_intercept'] = float(t_parts[0])
                    data['t_slope'] = float(t_parts[1])
                elif line.startswith('F-value:'):
                    data['f_value'] = float(line.split(': ')[1])
                elif line.startswith('Residual Sum of Squares:'):
                    data['rss'] = float(line.split(': ')[1])
            
            results.append(data)
        except:
            pass
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('f_value', ascending=False)
        df.to_csv(os.path.join(output_path, 'regression_results.csv'), index=False)
        return df
    return None

def collect_binary_results(binary_regression_path, output_path):
    """收集二元回归分析结果"""
    txt_files = [f for f in os.listdir(binary_regression_path) if f.endswith('_summary.txt')]
    results = []
    
    for txt_file in txt_files:
        try:
            with open(os.path.join(binary_regression_path, txt_file), 'r') as f:
                content = f.read()
            
            feature_name = os.path.basename(txt_file).replace('_summary.txt', '')
            data = {'feature_name': feature_name}
            
            for line in content.split('\n'):
                if line.startswith('Sample size:'):
                    data['sample_size'] = int(line.split(': ')[1])
                elif line.startswith('Coefficients:'):
                    coef_str = line.split(': ')[1].strip('[]')
                    coef_parts = coef_str.split()
                    data['intercept'] = float(coef_parts[0])
                    data['slope'] = float(coef_parts[1])
                elif line.startswith('t-values:'):
                    t_str = line.split(': ')[1].strip('[]')
                    t_parts = t_str.split()
                    data['t_intercept'] = float(t_parts[0])
                    data['t_slope'] = float(t_parts[1])
                elif line.startswith('F-value:'):
                    data['f_value'] = float(line.split(': ')[1])
                elif line.startswith('Residual Sum of Squares:'):
                    data['rss'] = float(line.split(': ')[1])
            
            results.append(data)
        except:
            pass
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('f_value', ascending=False)
        df.to_csv(os.path.join(output_path, 'binary_regression_results.csv'), index=False)
        return df
    return None

def collect_statistics_results(stats_path, output_path):
    """收集统计指标结果"""
    csv_files = [f for f in os.listdir(stats_path) if f.endswith('_stats.csv')]
    results = []
    
    for csv_file in csv_files:
        try:
            feature_name = os.path.basename(csv_file).replace('_stats.csv', '')
            df = pd.read_csv(os.path.join(stats_path, csv_file))
            if not df.empty:
                stats_data = df.iloc[0].to_dict()
                stats_data['feature_name'] = feature_name
                results.append(stats_data)
        except:
            pass
    
    if results:
        df = pd.DataFrame(results)
        # 按绝对相关系数排序
        df['abs_label_corr'] = df['label_corr'].abs()
        df = df.sort_values('abs_label_corr', ascending=False)
        df.to_csv(os.path.join(output_path, 'statistics_results.csv'), index=False)
        return df
    return None

def main():
    parser = argparse.ArgumentParser(description="H5特征分析 - 增强版本")
    parser.add_argument("--feature_h5_path", type=str, 
                       default="/Users/wook/Downloads/FactorTest1/feature_test_data/feature/combined_features.h5")
    parser.add_argument("--label_h5_path", type=str, 
                       default="/Users/wook/Downloads/FactorTest1/feature_test_data/label/combined_labels.h5")
    parser.add_argument("--factor_name_path", type=str, 
                       default="/Users/wook/Downloads/FactorTest1/feature_error_test/factorname_26.csv")
    parser.add_argument("--output_base_path", type=str, 
                       default="/Users/wook/Downloads/FactorTest1/feature_analysis_h5")
    args = parser.parse_args()

    # 检查文件存在性
    if not os.path.exists(args.feature_h5_path):
        print(f"❌ 特征文件不存在: {args.feature_h5_path}")
        return
    
    if not os.path.exists(args.label_h5_path):
        print(f"❌ 标签文件不存在: {args.label_h5_path}")
        return
    
    try:
        # 处理特征
        process_features_from_h5(args.feature_h5_path, args.label_h5_path, 
                               args.output_base_path, args.factor_name_path)
        
        # 收集各种结果
        print("📊 收集分析结果...")
        
        # 线性回归结果
        regression_path = os.path.join(args.output_base_path, "labelregressionsummary")
        if os.path.exists(regression_path):
            collect_results(regression_path, args.output_base_path)
        
        # 二元回归结果
        binary_regression_path = os.path.join(args.output_base_path, "binaryregression")
        if os.path.exists(binary_regression_path):
            collect_binary_results(binary_regression_path, args.output_base_path)
        
        # 统计指标结果
        stats_path = os.path.join(args.output_base_path, "featurestat")
        if os.path.exists(stats_path):
            collect_statistics_results(stats_path, args.output_base_path)
        
        print(f"✅ 所有结果已保存到: {args.output_base_path}")
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()