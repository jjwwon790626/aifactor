import pandas as pd
import numpy as np
from scipy.stats import spearmanr, skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FactorAnalyzer:
    def __init__(self, base_path="/Users/wook/Downloads/FactorTest1"):
        self.base_path = Path(base_path)
        self.output_path = self.base_path / "ic_ccc_all"
        self.output_path.mkdir(exist_ok=True)
        
    def load_data(self):
        """加载H5数据并过滤标签为0的行 """
        print("加载数据...")
        
        # 使用pandas读取H5数据（与第一个代码完全一致）
        features_df = pd.read_hdf(self.base_path / "feature_test_data/feature/all_combined_features.h5", key='features')
        labels_df = pd.read_hdf(self.base_path / "feature_test_data/label/all_combined_labels.h5", key='labels')
        labels = labels_df['labels'].values
        
        # 获取特征名称（与第一个代码一致）
        feature_names = features_df.columns.tolist()
        
        # 读取因子名称文件（如果提供）
        factor_name_path = self.base_path / "feature_error_test/factorname_26.csv"
        if factor_name_path.exists():
            try:
                factor_names_df = pd.read_csv(factor_name_path, header=None)
                factor_names_from_file = factor_names_df[0].str.replace('.csv', '').tolist()
                if len(factor_names_from_file) == len(feature_names):
                    feature_names = factor_names_from_file
            except:
                pass
        
        self.factor_names = feature_names
        
        # 验证数据一致性（与第一个代码一致）
        if len(features_df) != len(labels):
            min_len = min(len(features_df), len(labels))
            features_df = features_df.iloc[:min_len]
            labels = labels[:min_len]
        
        # 过滤掉标签为0的行（与第一个代码完全一致）
        non_zero_mask = (labels != 0)
        features_df_filtered = features_df[non_zero_mask].copy()
        labels_filtered = labels[non_zero_mask].copy()
        
        # 转换数据类型（与第一个代码一致）
        for col in features_df_filtered.columns:
            features_df_filtered[col] = pd.to_numeric(features_df_filtered[col], errors='coerce')
        
        feature_matrix = features_df_filtered.astype(np.float64).values
        labels_filtered = labels_filtered.astype(np.float64)
        
        print(f"加载完成: {feature_matrix.shape[0]} 行, {len(self.factor_names)} 个特征 (已过滤标签为0的行)")
        
        # 重新创建DataFrame
        self.features_df = pd.DataFrame(feature_matrix, columns=self.factor_names)
        self.labels_df = pd.DataFrame(labels_filtered, columns=['return'])

    def ols_regression(self, y, X):
        """借鉴第一个代码的OLS回归实现 - 包含更严格的数据处理"""
        # 确保输入是NumPy数组
        y = np.asarray(y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64).flatten()
        
        # 数据截断处理（与第一个代码一致）
        y = np.clip(y, -1e8, 1e8)
        X = np.clip(X, -1e8, 1e8)
        
        # 移除无效值（与第一个代码的处理方式一致）
        try:
            feature_valid = ~(pd.isna(X) | np.isinf(X))
            label_valid = ~(pd.isna(y) | np.isinf(y))
            valid_mask = feature_valid & label_valid
            X = X[valid_mask]
            y = y[valid_mask]
        except:
            finite_mask = np.isfinite(X) & np.isfinite(y)
            X = X[finite_mask]
            y = y[finite_mask]
        
        # 检查是否有足够的数据点
        if len(X) < 2:
            return np.zeros(2), 0, np.zeros(2), 0
        
        # 添加截距项
        X_with_intercept = np.column_stack((np.ones(len(X)), X))
        
        try:
            # 计算回归系数 β = (X'X)^(-1)X'y
            XtX = np.dot(X_with_intercept.T, X_with_intercept)
            XtX_inv = np.linalg.inv(XtX)
            XtY = np.dot(X_with_intercept.T, y)
            beta = np.dot(XtX_inv, XtY)
            
            # 计算预测值和残差
            y_pred = np.dot(X_with_intercept, beta)
            rss = np.sum((y - y_pred) ** 2)  # 残差平方和
            tss = np.sum((y - np.mean(y)) ** 2)  # 总平方和
            ess = tss - rss  # 解释平方和
            
            # 计算统计量
            n, k = X_with_intercept.shape
            mse_model = ess / (k - 1) if (k - 1) > 0 else 0
            mse_residual = rss / (n - k) if (n - k) > 0 else 0
            F_value = mse_model / mse_residual if mse_residual > 0 else 0
            
            # 计算标准误差和t统计量
            se = np.sqrt(np.diag(XtX_inv) * mse_residual)
            t_values = beta / se if np.all(se > 0) else np.zeros(k)
            
            return beta, rss, t_values, F_value
            
        except (np.linalg.LinAlgError, ValueError):
            # 处理矩阵奇异或其他数值问题
            return np.zeros(2), 0, np.zeros(2), 0

    def concordance_correlation_coefficient(self, y_true, y_pred):
        """计算CCC"""
        mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
        var_true, var_pred = np.var(y_true), np.var(y_pred)
        covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
        return (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred)**2)

    def calculate_statistics(self, data):
        """计算统计量"""
        return {
            "max": np.max(data), "min": np.min(data), "mean": np.mean(data), "std": np.std(data),
            "90%": np.percentile(data, 90), "75%": np.percentile(data, 75), "50%": np.percentile(data, 50),
            "25%": np.percentile(data, 25), "10%": np.percentile(data, 10),
            "skew": skew(data), "kurt": kurtosis(data),
            "mean/std": np.mean(data) / np.std(data) if np.std(data) != 0 else np.nan
        }

    def calculate_all_metrics(self):
        """计算所有指标 - 使用改进的回归计算方法，移除二元回归"""
        print("计算指标...")
        results = []
        regression_results = []
        
        for factor in self.factor_names:
            try:
                # 获取因子数据
                feature_data = self.features_df[factor].values
                label_data = self.labels_df['return'].values
                
                # 借鉴第一个代码的数据清洗方式
                # 转换数据类型并进行截断处理
                feature_data = np.asarray(feature_data, dtype=np.float64)
                label_data = np.asarray(label_data, dtype=np.float64)
                
                # 数据截断处理
                feature_data = np.clip(feature_data, -1e8, 1e8)
                label_data = np.clip(label_data, -1e8, 1e8)
                
                # 移除无效值（采用第一个代码的处理方式）
                try:
                    feature_valid = ~(pd.isna(feature_data) | np.isinf(feature_data))
                    label_valid = ~(pd.isna(label_data) | np.isinf(label_data))
                    valid_mask = feature_valid & label_valid
                    feature_clean = feature_data[valid_mask]
                    label_clean = label_data[valid_mask]
                except:
                    finite_mask = np.isfinite(feature_data) & np.isfinite(label_data)
                    feature_clean = feature_data[finite_mask]
                    label_clean = label_data[finite_mask]
                
                if len(feature_clean) > 10:
                    # IC和CCC计算
                    ic, _ = spearmanr(feature_clean, label_clean)
                    ccc = self.concordance_correlation_coefficient(feature_clean, label_clean)
                    
                    results.append({
                        'Factor': factor,
                        'IC': ic if not np.isnan(ic) else 0,
                        'CCC': ccc if not np.isnan(ccc) else 0,
                        'IC_abs': abs(ic) if not np.isnan(ic) else 0,
                        'CCC_abs': abs(ccc) if not np.isnan(ccc) else 0,
                        'sample_count': len(feature_clean)
                    })
                    
                    # 线性回归分析 - 使用改进的方法
                    try:
                        beta, rss, t_values, F_value = self.ols_regression(label_clean, feature_clean)
                        regression_results.append({
                            'feature_name': factor,
                            'sample_size': len(label_clean),
                            'intercept': beta[0],
                            'slope': beta[1],
                            'rss': rss,
                            't_intercept': t_values[0],
                            't_slope': t_values[1],
                            'f_value': F_value
                        })
                    except Exception as e:
                        print(f"回归计算错误 {factor}: {e}")
                        
            except Exception as e:
                print(f"因子处理错误 {factor}: {e}")
                
        self.metrics_df = pd.DataFrame(results)
        self.regression_df = pd.DataFrame(regression_results)
        
        # 计算IC和CCC的详细统计
        ic_stats, ccc_stats = [], []
        for _, row in self.metrics_df.iterrows():
            factor = row['Factor']
            feature_data = self.features_df[factor].dropna()
            label_data = self.labels_df['return'].loc[feature_data.index].dropna()
            
            if len(feature_data) > 100:
                # 滑动窗口计算
                window_size = min(1000, len(feature_data) // 5)
                step_size = max(100, window_size // 10)
                ic_samples, ccc_samples = [], []
                
                for start in range(0, len(feature_data) - window_size, step_size):
                    end = start + window_size
                    f_window = feature_data.iloc[start:end]
                    l_window = label_data.iloc[start:end]
                    
                    if len(f_window) > 10:
                        ic_temp, _ = spearmanr(f_window, l_window)
                        ccc_temp = self.concordance_correlation_coefficient(f_window, l_window)
                        if not np.isnan(ic_temp): ic_samples.append(ic_temp)
                        if not np.isnan(ccc_temp): ccc_samples.append(ccc_temp)
                
                if len(ic_samples) < 5:
                    ic_samples = [row['IC']] * 10
                    ccc_samples = [row['CCC']] * 10
                
                ic_stats.append({"Factor": factor, **self.calculate_statistics(ic_samples)})
                ccc_stats.append({"Factor": factor, **self.calculate_statistics(ccc_samples)})
            else:
                default_stats = {"Factor": factor, "max": 0, "min": 0, "mean": 0, "std": 0,
                               "90%": 0, "75%": 0, "50%": 0, "25%": 0, "10%": 0,
                               "skew": 0, "kurt": 0, "mean/std": 0}
                ic_stats.append(default_stats)
                ccc_stats.append(default_stats)
        
        self.ic_stats_df = pd.DataFrame(ic_stats)
        self.ccc_stats_df = pd.DataFrame(ccc_stats)

    def create_correlation_heatmap(self):
        """创建相关性热力图"""
        print("生成相关性热力图...")
        corr_matrix = self.features_df.corr()
        
        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', 
                   center=0, square=True, cbar_kws={"shrink": .8})
        plt.title('Factor Correlation Heatmap', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(self.output_path / 'factor_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        return corr_matrix

    def identify_redundant_factors(self, corr_matrix, threshold=0.8):
        """识别冗余因子"""
        redundant_pairs = []
        high_corr_matrix = corr_matrix.abs() > threshold
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if high_corr_matrix.iloc[i, j]:
                    factor1, factor2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    ic1 = self.metrics_df[self.metrics_df['Factor'] == factor1]['IC_abs'].iloc[0]
                    ic2 = self.metrics_df[self.metrics_df['Factor'] == factor2]['IC_abs'].iloc[0]
                    
                    redundant_pairs.append({
                        'Factor1': factor1, 'Factor2': factor2,
                        'Correlation': corr_matrix.iloc[i, j],
                        'IC1': ic1, 'IC2': ic2,
                        'Recommended_Remove': factor1 if ic1 < ic2 else factor2
                    })
        return pd.DataFrame(redundant_pairs)

    def factor_selection_strategy(self):
        """因子筛选策略"""
        ranked_factors = self.metrics_df.sort_values('IC_abs', ascending=False)
        top_tier = ranked_factors.head(int(len(ranked_factors) * 0.3))
        mid_tier = ranked_factors.iloc[int(len(ranked_factors) * 0.3):int(len(ranked_factors) * 0.7)]
        low_tier = ranked_factors.tail(int(len(ranked_factors) * 0.3))
        
        return {
            'Top_Tier_Factors': top_tier['Factor'].tolist(),
            'Mid_Tier_Factors': mid_tier['Factor'].tolist(),
            'Low_Tier_Factors': low_tier['Factor'].tolist(),
            'Recommended_Keep': top_tier['Factor'].tolist(),
            'Consider_Remove': low_tier['Factor'].tolist()
        }

    def create_comprehensive_summary(self):
        """创建综合汇总"""
        summary_data = []
        for factor in self.factor_names:
            metrics_row = self.metrics_df[self.metrics_df['Factor'] == factor]
            ic_stats_row = self.ic_stats_df[self.ic_stats_df['Factor'] == factor]
            ccc_stats_row = self.ccc_stats_df[self.ccc_stats_df['Factor'] == factor]
            regression_row = self.regression_df[self.regression_df['feature_name'] == factor]
            
            summary_row = {
                'Factor': factor,
                'IC': metrics_row['IC'].iloc[0] if not metrics_row.empty else 0,
                'IC_abs': metrics_row['IC_abs'].iloc[0] if not metrics_row.empty else 0,
                'IC_mean': ic_stats_row['mean'].iloc[0] if not ic_stats_row.empty else 0,
                'IC_std': ic_stats_row['std'].iloc[0] if not ic_stats_row.empty else 0,
                'IC_mean_std_ratio': ic_stats_row['mean/std'].iloc[0] if not ic_stats_row.empty else 0,
                'CCC': metrics_row['CCC'].iloc[0] if not metrics_row.empty else 0,
                'CCC_abs': metrics_row['CCC_abs'].iloc[0] if not metrics_row.empty else 0,
                'regression_slope': regression_row['slope'].iloc[0] if not regression_row.empty else 0,
                'regression_f_value': regression_row['f_value'].iloc[0] if not regression_row.empty else 0,
                'sample_count': metrics_row['sample_count'].iloc[0] if not metrics_row.empty else 0
            }
            summary_data.append(summary_row)
        
        comprehensive_df = pd.DataFrame(summary_data)
        comprehensive_df['IC_abs_rank'] = comprehensive_df['IC_abs'].rank(ascending=False)
        comprehensive_df['regression_f_value_rank'] = comprehensive_df['regression_f_value'].rank(ascending=False)
        
        # 综合评分
        comprehensive_df['composite_score'] = (
            0.4 * comprehensive_df['IC_abs_rank'] / len(comprehensive_df) +
            0.4 * comprehensive_df['regression_f_value_rank'] / len(comprehensive_df) +
            0.2 * comprehensive_df['IC_mean_std_ratio'].rank(ascending=False) / len(comprehensive_df)
        )
        comprehensive_df = comprehensive_df.sort_values('composite_score', ascending=False)
        comprehensive_df.to_csv(self.output_path / 'comprehensive_factor_summary.csv', index=False)

    def save_results(self, corr_matrix, redundant_factors, strategy):
        """保存结果 - 移除二元回归相关保存"""
        print("保存结果...")
        self.metrics_df.to_csv(self.output_path / 'factor_metrics.csv', index=False)
        self.ic_stats_df.to_csv(self.output_path / 'ic_stats.csv', index=False)
        self.ccc_stats_df.to_csv(self.output_path / 'ccc_stats.csv', index=False)
        
        if not self.regression_df.empty:
            self.regression_df.sort_values('f_value', ascending=False).to_csv(
                self.output_path / 'regression_results.csv', index=False)
        
        corr_matrix.to_csv(self.output_path / 'correlation_matrix.csv')
        
        if len(redundant_factors) > 0:
            redundant_factors.to_csv(self.output_path / 'redundant_factors.csv', index=False)
        
        strategy_df = pd.DataFrame({
            'Category': ['Top_Tier', 'Mid_Tier', 'Low_Tier'],
            'Factors': [str(strategy['Top_Tier_Factors']), 
                       str(strategy['Mid_Tier_Factors']), 
                       str(strategy['Low_Tier_Factors'])]
        })
        strategy_df.to_csv(self.output_path / 'selection_strategy.csv', index=False)
        self.create_comprehensive_summary()

    def generate_report(self, corr_matrix, redundant_factors, strategy):
        """生成报告 - 移除二元回归相关内容"""
        print("生成分析报告...")
        
        # 安全地获取统计信息
        avg_ic = self.metrics_df['IC'].mean()
        avg_ccc = self.metrics_df['CCC'].mean()
        avg_regression_f = self.regression_df['f_value'].mean() if not self.regression_df.empty else 0
        max_corr = corr_matrix.abs().max().max()
        
        # 获取顶级因子表格
        top_factors_table = self.metrics_df.nlargest(10, 'IC_abs')[['Factor', 'IC', 'CCC', 'sample_count']].to_string(index=False)
        
        # 获取回归分析表格
        regression_table = ""
        if not self.regression_df.empty:
            regression_table = self.regression_df.nlargest(10, 'f_value')[['feature_name', 'slope', 'f_value']].to_string(index=False)
        else:
            regression_table = "无回归分析结果"
        
        # 获取推荐因子列表
        recommended_factors = ', '.join(strategy['Recommended_Keep'][:10])
        if len(strategy['Recommended_Keep']) > 10:
            recommended_factors += "..."
        
        report = f"""# 因子分析报告

## 1. 数据概览
- 总因子数量: {len(self.factor_names)}
- 有效样本数: {len(self.features_df)}

## 2. 顶级因子 (按IC绝对值排序)
{top_factors_table}

## 3. 回归分析最优因子
{regression_table}

## 4. 关键统计指标
- 平均IC: {avg_ic:.4f}
- 平均CCC: {avg_ccc:.4f}
- 平均回归F值: {avg_regression_f:.4f}
- 最大相关系数: {max_corr:.4f}
- 冗余因子对数: {len(redundant_factors)}

## 5. 推荐因子 ({len(strategy['Recommended_Keep'])}个)
{recommended_factors}

## 6. 结论
1. 优先使用IC绝对值高且回归F值大的因子
2. 移除{len(redundant_factors)}对高相关冗余因子
3. 重点关注综合评分排名前30%的因子
        """
        
        with open(self.output_path / 'factor_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)

    def run_analysis(self):
        """运行完整分析"""
        self.load_data()
        self.calculate_all_metrics()
        corr_matrix = self.create_correlation_heatmap()
        redundant_factors = self.identify_redundant_factors(corr_matrix)
        strategy = self.factor_selection_strategy()
        self.generate_report(corr_matrix, redundant_factors, strategy)
        self.save_results(corr_matrix, redundant_factors, strategy)
        
        # 安全地计算平均回归F值
        avg_regression_f = self.regression_df['f_value'].mean() if not self.regression_df.empty else 0
        
        print(f"\n✅ 分析完成！结果保存到: {self.output_path}")
        print(f"📊 有效样本: {len(self.features_df)}, 因子数: {len(self.factor_names)}")
        print(f"🎯 平均IC: {self.metrics_df['IC'].mean():.4f}")
        print(f"📈 平均回归F值: {avg_regression_f:.4f}")

# 运行分析
if __name__ == "__main__":
    analyzer = FactorAnalyzer()
    analyzer.run_analysis()