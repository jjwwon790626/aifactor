import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

# 添加 hft 库路径
hft_signals_path = '/Users/wook/Downloads/FactorTest/hft-signals'
if hft_signals_path not in sys.path:
    sys.path.insert(0, hft_signals_path)

# 导入HFT因子计算函数
from hft.signal.large_jump import large_jump
from hft.signal.llt import llt
from hft.signal.return_lag import ask_lag, bid_lag

# 导入精简版Alpha因子计算器
from hft.signal.SelectedAlphaFactorCalculator import SelectedAlphaFactorCalculator


class SimplifiedH5Processor:
    def __init__(self, snapshot_dir, trade_dir, output_paths):
        self.snapshot_dir = snapshot_dir
        self.trade_dir = trade_dir
        self.feature_path = output_paths['feature_path']
        self.label_path = output_paths['label_path']
        self.factor_name_path = output_paths['factor_name_path']
        self.Fre = 600
        
        # 只计算这5个HFT因子
        self.hft_factors = ['bid_lag', 'ask_lag', 'large_jump_up', 'llt', 'large_jump_down']
        
        # 创建输出目录
        for path in output_paths.values():
            os.makedirs(path, exist_ok=True)
        
        # 初始化累积数据
        self.all_factors_df = pd.DataFrame()
        self.all_labels = []
        self.factor_names_saved = False
        
        # 初始化Alpha因子计算器
        self.alpha_calculator = SelectedAlphaFactorCalculator()
    
    def get_h5_file_pairs(self):
        """获取配对的H5快照和交易文件"""
        snapshot_files = glob.glob(os.path.join(self.snapshot_dir, "*_snapshot.h5"))
        file_pairs = []
        
        for snapshot_file in snapshot_files:
            base_name = os.path.basename(snapshot_file).replace('_snapshot.h5', '')
            trade_file = os.path.join(self.trade_dir, base_name + '_trade.h5')
            
            if os.path.exists(trade_file):
                file_pairs.append((snapshot_file, trade_file, base_name))
        
        return sorted(file_pairs)
    
    def preprocess_data(self, ob):
        """数据预处理：转换为所需格式"""
        # 创建HFT格式数据（5档）
        hft_data = {}
        
        # 处理时间戳
        if 'timestamp' in ob.columns:
            hft_data['ts'] = ob['timestamp'].values
        else:
            hft_data['ts'] = ob.index.astype(np.int64) // 1000
        
        # 转换前5档为HFT格式
        for i in range(5):
            level = i + 1
            ask_price_col = f'asks[{i}].price'
            ask_amount_col = f'asks[{i}].amount'
            bid_price_col = f'bids[{i}].price'
            bid_amount_col = f'bids[{i}].amount'
            
            hft_data[f'ap{level}'] = ob[ask_price_col].values if ask_price_col in ob.columns else np.nan
            hft_data[f'av{level}'] = ob[ask_amount_col].values if ask_amount_col in ob.columns else np.nan
            hft_data[f'bp{level}'] = ob[bid_price_col].values if bid_price_col in ob.columns else np.nan
            hft_data[f'bv{level}'] = ob[bid_amount_col].values if bid_amount_col in ob.columns else np.nan
        
        hft_ob = pd.DataFrame(hft_data).dropna()
        
        # 创建Alpha格式数据（25档）
        alpha_data = {}
        
        # 检查原始数据中有多少档
        max_available_level = 0
        for i in range(25):
            if f'asks[{i}].price' in ob.columns:
                max_available_level = i + 1
        
        # 转换已有的档位数据
        for i in range(max_available_level):
            ask_price_col = f'asks[{i}].price'
            ask_amount_col = f'asks[{i}].amount'
            bid_price_col = f'bids[{i}].price'
            bid_amount_col = f'bids[{i}].amount'
            
            alpha_data[f'ap{i}'] = ob[ask_price_col].values if ask_price_col in ob.columns else np.nan
            alpha_data[f'av{i}'] = ob[ask_amount_col].values if ask_amount_col in ob.columns else np.nan
            alpha_data[f'bp{i}'] = ob[bid_price_col].values if bid_price_col in ob.columns else np.nan
            alpha_data[f'bv{i}'] = ob[bid_amount_col].values if bid_amount_col in ob.columns else np.nan
        
        # 如果数据不足25档，填充缺失的档位
        if max_available_level < 25:
            if max_available_level > 0:
                last_idx = max_available_level - 1
                last_bp = alpha_data[f'bp{last_idx}']
                last_ap = alpha_data[f'ap{last_idx}']
                last_bv = alpha_data[f'bv{last_idx}']
                last_av = alpha_data[f'av{last_idx}']
                
                for i in range(max_available_level, 25):
                    price_decay_factor = 1 - 0.0005 * (i - last_idx)
                    price_growth_factor = 1 + 0.0005 * (i - last_idx)
                    volume_decay_factor = 0.8 ** (i - last_idx)
                    
                    alpha_data[f'bp{i}'] = last_bp * price_decay_factor
                    alpha_data[f'ap{i}'] = last_ap * price_growth_factor
                    alpha_data[f'bv{i}'] = last_bv * volume_decay_factor
                    alpha_data[f'av{i}'] = last_av * volume_decay_factor
            else:
                # 用默认值填充
                for i in range(25):
                    alpha_data[f'bp{i}'] = np.full(len(ob), 100.0)
                    alpha_data[f'ap{i}'] = np.full(len(ob), 101.0)
                    alpha_data[f'bv{i}'] = np.full(len(ob), 1.0)
                    alpha_data[f'av{i}'] = np.full(len(ob), 1.0)
        
        alpha_ob = pd.DataFrame(alpha_data).ffill().bfill().fillna(0)
        
        # 确保价格和成交量都是正数
        for i in range(25):
            alpha_ob[f'bp{i}'] = alpha_ob[f'bp{i}'].abs()
            alpha_ob[f'ap{i}'] = alpha_ob[f'ap{i}'].abs()
            alpha_ob[f'bv{i}'] = alpha_ob[f'bv{i}'].abs()
            alpha_ob[f'av{i}'] = alpha_ob[f'av{i}'].abs()
        
        return hft_ob, alpha_ob
    
    def calculate_hft_factors(self, hft_ob):
        """计算5个HFT因子"""
        factors_data = {}
        
        try:
            factors_data['llt'] = llt(datas={"depth5": hft_ob}, params={"n": self.Fre})
        except:
            factors_data['llt'] = np.zeros(len(hft_ob))
        
        try:
            factors_data['bid_lag'] = bid_lag(datas={'depth5': hft_ob}, params={'n': self.Fre})
        except:
            factors_data['bid_lag'] = np.zeros(len(hft_ob))
        
        try:
            factors_data['ask_lag'] = ask_lag(datas={'depth5': hft_ob}, params={'n': self.Fre})
        except:
            factors_data['ask_lag'] = np.zeros(len(hft_ob))
        
        try:
            factors_data['large_jump_up'] = large_jump(datas={'depth5': hft_ob}, params={'n': self.Fre, 'direct': 'up', 'jump_ratio': 0.001})
        except:
            factors_data['large_jump_up'] = np.zeros(len(hft_ob))
        
        try:
            factors_data['large_jump_down'] = large_jump(datas={'depth5': hft_ob}, params={'n': 100, 'direct': 'down', 'jump_ratio': 0.001})
        except:
            factors_data['large_jump_down'] = np.zeros(len(hft_ob))
        
        return pd.DataFrame(factors_data)
    
    def calculate_labels(self, hft_ob):
        """计算标签"""
        mid_price = (hft_ob['ap1'] + hft_ob['bp1']) / 2
        labels = mid_price.pct_change(self.Fre).shift(-self.Fre).fillna(0)
        return labels.values
    
    def save_factor_names(self, factor_names):
        """保存因子名称（只保存一次）"""
        if not self.factor_names_saved:
            factor_names_list = [f"{name}.csv" for name in factor_names]
            factor_names_df = pd.DataFrame(factor_names_list, columns=['factor_name'])
            factor_names_path = os.path.join(self.factor_name_path, "factorname_selected.csv")
            factor_names_df.to_csv(factor_names_path, header=False, index=False)
            self.factor_names_saved = True
    
    def process_single_file_pair(self, snapshot_file, trade_file, base_name):
        """处理单个文件对"""
        try:
            # 读取H5数据
            ob_raw = pd.read_hdf(snapshot_file, key='snapshot')
            
            # 数据预处理
            hft_ob, alpha_ob = self.preprocess_data(ob_raw)
            
            if len(hft_ob) == 0 or len(alpha_ob) == 0:
                return
            
            # 计算HFT因子
            hft_factors_df = self.calculate_hft_factors(hft_ob)
            
            # 计算Alpha因子
            alpha_factors_df = self.alpha_calculator.calculate_all_factors(alpha_ob)
            
            # 合并因子
            min_length = min(len(hft_factors_df), len(alpha_factors_df))
            if min_length == 0:
                return
                
            hft_factors_df = hft_factors_df.iloc[:min_length].reset_index(drop=True)
            alpha_factors_df = alpha_factors_df.iloc[:min_length].reset_index(drop=True)
            
            combined_factors_df = pd.concat([hft_factors_df, alpha_factors_df], axis=1).fillna(0)
            
            # 计算标签
            labels = self.calculate_labels(hft_ob)[:min_length]
            
            # 累积数据
            if self.all_factors_df.empty:
                self.all_factors_df = combined_factors_df.copy()
                self.all_labels = labels.tolist()
                self.save_factor_names(combined_factors_df.columns)
            else:
                if set(combined_factors_df.columns) == set(self.all_factors_df.columns):
                    self.all_factors_df = pd.concat([self.all_factors_df, combined_factors_df], ignore_index=True)
                    self.all_labels.extend(labels.tolist())
            
        except Exception as e:
            print(f"处理文件 {base_name} 时出错: {e}")
    
    def save_final_data(self):
        """保存最终数据"""
        # 保存H5格式
        h5_feature_path = os.path.join(self.feature_path, "selected_features.h5")
        self.all_factors_df.to_hdf(h5_feature_path, key='features', mode='w', format='table', complevel=9)
        
        h5_label_path = os.path.join(self.label_path, "selected_labels.h5")
        pd.DataFrame(self.all_labels, columns=['labels']).to_hdf(h5_label_path, key='labels', mode='w', format='table', complevel=9)
        
        # 保存CSV格式
        csv_feature_path = os.path.join(self.feature_path, "selected_features.csv")
        self.all_factors_df.to_csv(csv_feature_path, index=False)
        
        csv_label_path = os.path.join(self.label_path, "selected_labels.csv")
        pd.DataFrame(self.all_labels, columns=['labels']).to_csv(csv_label_path, index=False)
    
    def process_all_files(self):
        """处理所有文件"""
        file_pairs = self.get_h5_file_pairs()
        
        if not file_pairs:
            print("未找到任何配对的H5数据文件！")
            return
        
        print(f"🎯 计算因子:")
        print(f"HFT因子: {', '.join(self.hft_factors)}")
        print(f"Alpha因子: {len(self.alpha_calculator.target_factors)}个")
        print()
        
        with tqdm(total=len(file_pairs), desc="处理文件", unit="文件") as pbar:
            for snapshot_file, trade_file, base_name in file_pairs:
                pbar.set_description(f"处理: {os.path.basename(base_name)}")
                self.process_single_file_pair(snapshot_file, trade_file, base_name)
                pbar.update(1)
        
        # 保存最终数据
        if not self.all_factors_df.empty:
            self.save_final_data()
            print(f"\n✅ 处理完成!")
            print(f"总数据点: {len(self.all_factors_df)}")
            print(f"计算的因子数: {len(self.all_factors_df.columns)}")
            print(f"HFT因子: {[col for col in self.all_factors_df.columns if col in self.hft_factors]}")
            print(f"Alpha因子数: {len([col for col in self.all_factors_df.columns if col.startswith('factor_')])}")
        else:
            print("❌ 没有成功处理任何数据文件！")


def main():
    """主函数"""
    # 设置路径
    snapshot_dir = "/Users/wook/Downloads/因子挖掘/Sample data/raw/筛选BTC分段数据_dollarbar/book25_snapshot_1s"
    trade_dir = "/Users/wook/Downloads/因子挖掘/Sample data/raw/筛选BTC分段数据_dollarbar/trade"
    
    output_paths = {
        'feature_path': "/Users/wook/Downloads/FactorTest1/feature_test_data/feature",
        'label_path': "/Users/wook/Downloads/FactorTest1/feature_test_data/label",
        'factor_name_path': "/Users/wook/Downloads/FactorTest1/feature_error_test"
    }
    
    print("🚀 开始处理H5数据文件...")
    print("🎯 只计算指定的因子: 5个HFT因子 + 23个Alpha因子")
    
    # 创建处理器并运行
    processor = SimplifiedH5Processor(snapshot_dir, trade_dir, output_paths)
    processor.process_all_files()


if __name__ == "__main__":
    main()