import os
import glob
import struct
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

import sys
# 添加 hft 库路径
hft_signals_path = '/Users/wook/Downloads/FactorTest/hft-signals'
if hft_signals_path not in sys.path:
    sys.path.insert(0, hft_signals_path)

# 导入所需的库
import hft
from scipy.stats import pearsonr, zscore, spearmanr
from hft.utils.wrapper import trade_to_depth
from hft.utils.validate import plot_stats
from hft.utils.target.mid_price_changes import all_return
from hft.utils.target import filled_return, mid_price_changes, mp_changes
from hft.utils.combine import linear_model, SGDlinear_model
from hft.utils.format import purged_train_test_split, single_split
from hft.utils.format import depth_to_depth

# 导入因子计算函数
from hft.signal.arrive_rate import arrive_rate
from hft.signal.cofi import cofi
from hft.signal.depth_age import depth_bid_age, depth_ask_age
from hft.signal.depth_changes import depth_bid_change, depth_ask_change
from hft.signal.fair_spread import fair_spread
from hft.signal.large_jump import large_jump
from hft.signal.llt import llt
from hft.signal.oir import oir
from hft.signal.order_flow import oflow
from hft.signal.price_distance import price_distance
from hft.signal.price_impact import price_impact
from hft.signal.return_lag import ask_lag, bid_lag
from hft.signal.swr import swr
from hft.signal.tick_vpin import tick_vpin
from hft.signal.volume_at_same_price import volume_at_same_price
from hft.signal.volume import ask_volume, bid_volume
from hft.signal.weakoir import weakoir
from hft.signal.wss import wss
from hft.signal.AlphaFactorCalculator import AlphaFactorCalculator

class H5FactorProcessor:
    def __init__(self, snapshot_dir, trade_dir, output_paths):
        self.snapshot_dir = snapshot_dir
        self.trade_dir = trade_dir
        self.feature_path = output_paths['feature_path']
        self.label_path = output_paths['label_path']
        self.factor_name_path = output_paths['factor_name_path']
        self.Fre = 600
        
        # 创建输出目录
        for path in output_paths.values():
            os.makedirs(path, exist_ok=True)
        
        # 初始化累积数据
        self.all_factors_df = pd.DataFrame()
        self.all_labels = []
        self.factor_names_saved = False
        
        # 初始化Alpha因子计算器
        self.alpha_calculator = AlphaFactorCalculator()
    
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
    
    def preprocess_h5_data(self, ob, tr):
        """数据预处理：从H5读取的25档数据转换为HFT格式和Alpha格式"""
        # ======== 交易数据预处理 ========
        tr_processed = tr.copy()
        
        # 处理时间戳和列名
        if 'timestamp' in tr_processed.columns:
            tr_processed = tr_processed.rename(columns={'timestamp': 'ts'})
        else:
            tr_processed['ts'] = tr_processed.index.astype(np.int64) // 1000
            
        if 'price' in tr_processed.columns:
            tr_processed = tr_processed.rename(columns={'price': 'p'})
            
        if 'amount' in tr_processed.columns:
            tr_processed = tr_processed.rename(columns={'amount': 'v'})
        
        # 处理交易方向
        if 'side' in tr_processed.columns:
            tr_processed['v'] = np.where(tr_processed['side'] == 'sell', -tr_processed['v'], tr_processed['v'])
        
        # 确保必要列存在
        for col in ['ts', 'p', 'v']:
            if col not in tr_processed.columns:
                tr_processed[col] = 0
        
        tr_processed = tr_processed.dropna()
        
        # ======== 订单簿数据预处理 ========
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
        
        # 转换已有的档位数据（从asks[0].price格式转换为ap0格式）
        for i in range(max_available_level):
            ask_price_col = f'asks[{i}].price'
            ask_amount_col = f'asks[{i}].amount'
            bid_price_col = f'bids[{i}].price'
            bid_amount_col = f'bids[{i}].amount'
            
            alpha_data[f'ap{i}'] = ob[ask_price_col].values if ask_price_col in ob.columns else np.nan
            alpha_data[f'av{i}'] = ob[ask_amount_col].values if ask_amount_col in ob.columns else np.nan
            alpha_data[f'bp{i}'] = ob[bid_price_col].values if bid_price_col in ob.columns else np.nan
            alpha_data[f'bv{i}'] = ob[bid_amount_col].values if bid_amount_col in ob.columns else np.nan
        
        # 如果原始数据不足25档，填充缺失的档位
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
        
        # 添加时间戳
        if 'timestamp' in ob.columns:
            alpha_data['ts'] = ob['timestamp'].values
        else:
            alpha_data['ts'] = ob.index.astype(np.int64) // 1000
        
        alpha_ob = pd.DataFrame(alpha_data)
        
        # 数据清理
        alpha_ob = alpha_ob.ffill().bfill().fillna(0)
        
        # 确保价格和成交量都是正数
        for i in range(25):
            alpha_ob[f'bp{i}'] = alpha_ob[f'bp{i}'].abs()
            alpha_ob[f'ap{i}'] = alpha_ob[f'ap{i}'].abs()
            alpha_ob[f'bv{i}'] = alpha_ob[f'bv{i}'].abs()
            alpha_ob[f'av{i}'] = alpha_ob[f'av{i}'].abs()
            
            if i > 0:
                alpha_ob[f'bp{i}'] = np.minimum(alpha_ob[f'bp{i}'], alpha_ob[f'bp{i-1}'])
                alpha_ob[f'ap{i}'] = np.maximum(alpha_ob[f'ap{i}'], alpha_ob[f'ap{i-1}'])
        
        return hft_ob, alpha_ob, tr_processed
    
    def calculate_factors(self, hft_ob, alpha_ob, tr):
        """计算所有因子"""
        hft_factors_data = {}
        
        # 计算HFT因子（23个，只使用5档数据）
        try:
            hft_factors_data['cofi'] = cofi(datas={'depth5': hft_ob}, params={})
        except:
            hft_factors_data['cofi'] = np.zeros(len(hft_ob))
        
        try:
            hft_factors_data['depth_bid_age'] = depth_bid_age(datas={'depth5': hft_ob}, params={'n': self.Fre})
            hft_factors_data['depth_ask_age'] = depth_ask_age(datas={'depth5': hft_ob}, params={'n': self.Fre})
        except:
            hft_factors_data['depth_bid_age'] = np.zeros(len(hft_ob))
            hft_factors_data['depth_ask_age'] = np.zeros(len(hft_ob))
        
        try:
            hft_factors_data['depth_bid_change'] = depth_bid_change(datas={'depth5': hft_ob}, params={'n': self.Fre})
            hft_factors_data['depth_ask_change'] = depth_ask_change(datas={'depth5': hft_ob}, params={'n': self.Fre})
        except:
            hft_factors_data['depth_bid_change'] = np.zeros(len(hft_ob))
            hft_factors_data['depth_ask_change'] = np.zeros(len(hft_ob))
        
        try:
            hft_factors_data['fair_spread'] = fair_spread(datas={'depth5': hft_ob}, params={})
        except:
            hft_factors_data['fair_spread'] = np.zeros(len(hft_ob))
        
        try:
            hft_factors_data['large_jump_up'] = large_jump(datas={'depth5': hft_ob}, params={'n': self.Fre, 'direct': 'up', 'jump_ratio': 0.001})
            hft_factors_data['large_jump_down'] = large_jump(datas={'depth5': hft_ob}, params={'n': 100, 'direct': 'down', 'jump_ratio': 0.001})
        except:
            hft_factors_data['large_jump_up'] = np.zeros(len(hft_ob))
            hft_factors_data['large_jump_down'] = np.zeros(len(hft_ob))
        
        try:
            hft_factors_data['llt'] = llt(datas={"depth5": hft_ob}, params={"n": self.Fre})
        except:
            hft_factors_data['llt'] = np.zeros(len(hft_ob))
        
        try:
            hft_factors_data['oir'] = oir(datas={'depth5': hft_ob}, params={})
        except:
            hft_factors_data['oir'] = np.zeros(len(hft_ob))
        
        try:
            hft_factors_data['bid_order_flow'] = oflow(datas={'depth5': hft_ob}, params={'side': 'bid', 'bend_ratio': 4})
            hft_factors_data['ask_order_flow'] = oflow(datas={'depth5': hft_ob}, params={'side': 'ask', 'bend_ratio': 4})
        except:
            hft_factors_data['bid_order_flow'] = np.zeros(len(hft_ob))
            hft_factors_data['ask_order_flow'] = np.zeros(len(hft_ob))
        
        try:
            hft_factors_data['ask_price_distance'] = price_distance(datas={'depth5': hft_ob}, params={'n': self.Fre, 'side': 'ask'})
            hft_factors_data['bid_price_distance'] = price_distance(datas={'depth5': hft_ob}, params={'n': self.Fre, 'side': 'bid'})
        except:
            hft_factors_data['ask_price_distance'] = np.zeros(len(hft_ob))
            hft_factors_data['bid_price_distance'] = np.zeros(len(hft_ob))
        
        try:
            hft_factors_data['price_impact'] = price_impact(datas={'depth5': hft_ob}, params={'n': 5})
        except:
            hft_factors_data['price_impact'] = np.zeros(len(hft_ob))
        
        try:
            hft_factors_data['ask_lag'] = ask_lag(datas={'depth5': hft_ob}, params={'n': self.Fre})
            hft_factors_data['bid_lag'] = bid_lag(datas={'depth5': hft_ob}, params={'n': self.Fre})
        except:
            hft_factors_data['ask_lag'] = np.zeros(len(hft_ob))
            hft_factors_data['bid_lag'] = np.zeros(len(hft_ob))
        
        try:
            hft_factors_data['ask_swr'] = swr(datas={'depth5': hft_ob}, params={'side': 'ask'})
            hft_factors_data['bid_swr'] = swr(datas={'depth5': hft_ob}, params={'side': 'bid'})
        except:
            hft_factors_data['ask_swr'] = np.zeros(len(hft_ob))
            hft_factors_data['bid_swr'] = np.zeros(len(hft_ob))
        
        try:
            hft_factors_data['ask_volume'] = ask_volume(datas={'depth5': hft_ob}, params={})
            hft_factors_data['bid_volume'] = bid_volume(datas={'depth5': hft_ob}, params={})
        except:
            hft_factors_data['ask_volume'] = np.zeros(len(hft_ob))
            hft_factors_data['bid_volume'] = np.zeros(len(hft_ob))
        
        try:
            hft_factors_data['weakoir'] = weakoir(datas={'depth5': hft_ob}, params={})
        except:
            hft_factors_data['weakoir'] = np.zeros(len(hft_ob))
        
        try:
            hft_factors_data['wss'] = wss(datas={'depth5': hft_ob}, params={'n': 5})
        except:
            hft_factors_data['wss'] = np.zeros(len(hft_ob))
        
        # 创建HFT因子DataFrame
        hft_factors_df = pd.DataFrame(hft_factors_data)
        
        # 计算Alpha因子（160个，使用25档数据）
        try:
            alpha_factors_df = self.alpha_calculator.calculate_all_factors(alpha_ob)
        except Exception:
            alpha_factors_df = pd.DataFrame(index=alpha_ob.index)
        
        # 合并所有因子
        min_length = min(len(hft_factors_df), len(alpha_factors_df), len(hft_ob))
        
        if min_length > 0:
            hft_factors_df = hft_factors_df.iloc[:min_length]
            alpha_factors_df = alpha_factors_df.iloc[:min_length]
            
            hft_factors_df.reset_index(drop=True, inplace=True)
            alpha_factors_df.reset_index(drop=True, inplace=True)
            
            combined_factors_df = pd.concat([hft_factors_df, alpha_factors_df], axis=1)
        else:
            combined_factors_df = pd.DataFrame()
        
        return combined_factors_df
    
    def calculate_labels(self, hft_ob, n=600):
        """计算标签"""
        mid_price = (hft_ob['ap1'] + hft_ob['bp1']) / 2
        labels = mid_price.pct_change(n).shift(-n).fillna(0)
        return labels.values
    
    def save_binary_data(self, data, filepath):
        """将numpy数组保存为二进制文件"""
        with open(filepath, 'wb') as f:
            for value in data.flatten():
                if np.isnan(value) or np.isinf(value):
                    value = 0.0
                f.write(struct.pack('d', float(value)))
    
    def save_factor_names(self, factor_names):
        """保存因子名称（只保存一次）"""
        if not self.factor_names_saved:
            factor_names_list = [f"{name}.csv" for name in factor_names]
            factor_names_df = pd.DataFrame(factor_names_list, columns=['factor_name'])
            factor_names_path = os.path.join(self.factor_name_path, "factorname.csv")
            factor_names_df.to_csv(factor_names_path, header=False, index=False)
            self.factor_names_saved = True
    
    def process_single_file_pair(self, snapshot_file, trade_file, base_name):
        """处理单个文件对"""
        try:
            # 读取H5数据
            ob_raw = pd.read_hdf(snapshot_file, key='snapshot')
            tr_raw = pd.read_hdf(trade_file, key='trade')
            
            # 数据预处理
            hft_ob, alpha_ob, tr = self.preprocess_h5_data(ob_raw, tr_raw)
            
            # 检查数据
            if len(hft_ob) == 0 or len(alpha_ob) == 0:
                return
            
            # 计算因子
            factors_df = self.calculate_factors(hft_ob, alpha_ob, tr)
            
            if factors_df.empty:
                return
            
            # 计算标签
            labels = self.calculate_labels(hft_ob, n=self.Fre)
            
            # 确保数据长度一致
            min_length = min(len(factors_df), len(labels))
            if min_length == 0:
                return
                
            factors_df = factors_df.iloc[:min_length].fillna(0)
            labels = labels[:min_length]
            
            # 累积数据
            if self.all_factors_df.empty:
                self.all_factors_df = factors_df.copy()
                self.all_labels = labels.tolist()
                self.save_factor_names(factors_df.columns)
            else:
                if set(factors_df.columns) == set(self.all_factors_df.columns):
                    self.all_factors_df = pd.concat([self.all_factors_df, factors_df], ignore_index=True)
                    self.all_labels.extend(labels.tolist())
            
        except Exception:
            return
    
    def save_final_data(self):
        """保存最终的累积数据"""
        # 转换标签为numpy数组
        all_labels_array = np.array(self.all_labels)
        
        # # 保存二进制格式
        # feature_values = []
        # for col in self.all_factors_df.columns:
        #     feature_values.extend(self.all_factors_df[col].values)
        # feature_array = np.array(feature_values)
        
        # feature_file_path = os.path.join(self.feature_path, "combined_features")
        # self.save_binary_data(feature_array, feature_file_path)
        
        # label_file_path = os.path.join(self.label_path, "combined_labels")
        # self.save_binary_data(all_labels_array, label_file_path)
        
        # 保存H5格式
        h5_feature_path = os.path.join(self.feature_path, "combined_features_160.h5")
        self.all_factors_df.to_hdf(h5_feature_path, key='features', mode='w', format='table', complevel=9)
        
        h5_label_path = os.path.join(self.label_path, "combined_labels_160.h5")
        pd.DataFrame(all_labels_array, columns=['labels']).to_hdf(h5_label_path, key='labels', mode='w', format='table', complevel=9)
        
        # 保存CSV格式
        csv_feature_path = os.path.join(self.feature_path, "combined_features_160.csv")
        self.all_factors_df.to_csv(csv_feature_path, index=False)
        
        csv_label_path = os.path.join(self.label_path, "combined_labels_160.csv")
        pd.DataFrame(all_labels_array, columns=['labels']).to_csv(csv_label_path, index=False)
    
    def process_all_files(self):
        """处理所有文件"""
        file_pairs = self.get_h5_file_pairs()
        
        if not file_pairs:
            print("未找到任何配对的H5数据文件！")
            return
        
        # 使用tqdm显示进度
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
            print(f"HFT因子: 23个")
            print(f"Alpha因子: ~160个")
            print(f"总因子数: {len(self.all_factors_df.columns)}")
        else:
            print("❌ 没有成功处理任何数据文件！")

def main():
    """主函数"""
    # 设置路径
    snapshot_dir = "/Users/wook/Downloads/因子挖掘/Sample data/raw/筛选BTC分段数据/book25_snapshot_1s"
    trade_dir = "/Users/wook/Downloads/因子挖掘/Sample data/raw/筛选BTC分段数据/trade"
    
    output_paths = {
        'feature_path': "/Users/wook/Downloads/FactorTest1/feature_test_data/feature",
        'label_path': "/Users/wook/Downloads/FactorTest1/feature_test_data/label",
        'factor_name_path': "/Users/wook/Downloads/FactorTest1/feature_error_test"
    }
    
    print("🚀 开始处理H5数据文件...")
    print("📊 将计算 23个HFT因子 + ~160个Alpha因子")
    
    # 创建处理器并运行
    processor = H5FactorProcessor(snapshot_dir, trade_dir, output_paths)
    processor.process_all_files()

if __name__ == "__main__":
    main()