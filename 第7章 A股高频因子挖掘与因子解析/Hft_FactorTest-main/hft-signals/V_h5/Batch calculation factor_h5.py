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

class H5FactorProcessor:
    def __init__(self, snapshot_dir, trade_dir, output_paths):
        self.snapshot_dir = snapshot_dir
        self.trade_dir = trade_dir
        self.feature_path = output_paths['feature_path']
        self.label_path = output_paths['label_path']
        self.factor_name_path = output_paths['factor_name_path']
        self.Fre = 600  # 频率参数
        
        # 创建输出目录
        for path in output_paths.values():
            os.makedirs(path, exist_ok=True)
        
        # 初始化累积数据
        self.all_factors_df = pd.DataFrame()
        self.all_labels = []
        self.factor_names_saved = False
    
    def get_h5_file_pairs(self):
        """获取配对的H5快照和交易文件"""
        snapshot_files = glob.glob(os.path.join(self.snapshot_dir, "*_snapshot.h5"))
        file_pairs = []
        
        for snapshot_file in snapshot_files:
            # 提取基础文件名（去掉_snapshot.h5）
            base_name = os.path.basename(snapshot_file).replace('_snapshot.h5', '')
            trade_file = os.path.join(self.trade_dir, base_name + '_trade.h5')
            
            if os.path.exists(trade_file):
                file_pairs.append((snapshot_file, trade_file, base_name))
            else:
                print(f"警告: 找不到对应的交易文件: {trade_file}")
        
        return sorted(file_pairs)
    
    def convert_to_depth5(self, ob_df):
        """将25档数据转换为5档数据，并重命名列"""
        # 创建5档数据字典
        depth5_data = {
            'ts': ob_df['timestamp'].values if 'timestamp' in ob_df.columns else ob_df.index.astype(np.int64) // 1000
        }
        
        # 添加5档买卖盘数据
        for i in range(5):
            level = i + 1
            ask_price_col = f'asks[{i}].price'
            ask_amount_col = f'asks[{i}].amount'
            bid_price_col = f'bids[{i}].price'
            bid_amount_col = f'bids[{i}].amount'
            
            depth5_data[f'ap{level}'] = ob_df[ask_price_col].values if ask_price_col in ob_df.columns else np.nan
            depth5_data[f'av{level}'] = ob_df[ask_amount_col].values if ask_amount_col in ob_df.columns else np.nan
            depth5_data[f'bp{level}'] = ob_df[bid_price_col].values if bid_price_col in ob_df.columns else np.nan
            depth5_data[f'bv{level}'] = ob_df[bid_amount_col].values if bid_amount_col in ob_df.columns else np.nan
        
        # 创建DataFrame并移除NaN行
        depth5_df = pd.DataFrame(depth5_data).dropna()
        return depth5_df
    
    def process_trade_data(self, tr_df):
        """处理交易数据"""
        tr_processed = tr_df.copy()
        
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
        
        return tr_processed.dropna()
    
    def calculate_factors(self, ob, tr):
        """计算所有因子"""
        factors_dict = {}
        
        # 只计算基于订单簿的因子，避免索引匹配问题
        try:
            factors_dict['cofi'] = cofi(datas={'depth5': ob}, params={})
        except:
            factors_dict['cofi'] = np.zeros(len(ob))
        
        try:
            factors_dict['depth_bid_age'] = depth_bid_age(datas={'depth5': ob}, params={'n': self.Fre})
            factors_dict['depth_ask_age'] = depth_ask_age(datas={'depth5': ob}, params={'n': self.Fre})
        except:
            factors_dict['depth_bid_age'] = np.zeros(len(ob))
            factors_dict['depth_ask_age'] = np.zeros(len(ob))
        
        try:
            factors_dict['depth_bid_change'] = depth_bid_change(datas={'depth5': ob}, params={'n': self.Fre})
            factors_dict['depth_ask_change'] = depth_ask_change(datas={'depth5': ob}, params={'n': self.Fre})
        except:
            factors_dict['depth_bid_change'] = np.zeros(len(ob))
            factors_dict['depth_ask_change'] = np.zeros(len(ob))
        
        try:
            factors_dict['fair_spread'] = fair_spread(datas={'depth5': ob}, params={})
        except:
            factors_dict['fair_spread'] = np.zeros(len(ob))
        
        try:
            factors_dict['large_jump_up'] = large_jump(datas={'depth5': ob}, params={'n': self.Fre, 'direct': 'up', 'jump_ratio': 0.001})
            factors_dict['large_jump_down'] = large_jump(datas={'depth5': ob}, params={'n': 100, 'direct': 'down', 'jump_ratio': 0.001})
        except:
            factors_dict['large_jump_up'] = np.zeros(len(ob))
            factors_dict['large_jump_down'] = np.zeros(len(ob))
        
        try:
            factors_dict['llt'] = llt(datas={"depth5": ob}, params={"n": self.Fre})
        except:
            factors_dict['llt'] = np.zeros(len(ob))
        
        try:
            factors_dict['oir'] = oir(datas={'depth5': ob}, params={})
        except:
            factors_dict['oir'] = np.zeros(len(ob))
        
        try:
            factors_dict['bid_order_flow'] = oflow(datas={'depth5': ob}, params={'side': 'bid', 'bend_ratio': 4})
            factors_dict['ask_order_flow'] = oflow(datas={'depth5': ob}, params={'side': 'ask', 'bend_ratio': 4})
        except:
            factors_dict['bid_order_flow'] = np.zeros(len(ob))
            factors_dict['ask_order_flow'] = np.zeros(len(ob))
        
        try:
            factors_dict['ask_price_distance'] = price_distance(datas={'depth5': ob}, params={'n': self.Fre, 'side': 'ask'})
            factors_dict['bid_price_distance'] = price_distance(datas={'depth5': ob}, params={'n': self.Fre, 'side': 'bid'})
        except:
            factors_dict['ask_price_distance'] = np.zeros(len(ob))
            factors_dict['bid_price_distance'] = np.zeros(len(ob))
        
        try:
            factors_dict['price_impact'] = price_impact(datas={'depth5': ob}, params={'n': 5})
        except:
            factors_dict['price_impact'] = np.zeros(len(ob))
        
        try:
            factors_dict['ask_lag'] = ask_lag(datas={'depth5': ob}, params={'n': self.Fre})
            factors_dict['bid_lag'] = bid_lag(datas={'depth5': ob}, params={'n': self.Fre})
        except:
            factors_dict['ask_lag'] = np.zeros(len(ob))
            factors_dict['bid_lag'] = np.zeros(len(ob))
        
        try:
            factors_dict['ask_swr'] = swr(datas={'depth5': ob}, params={'side': 'ask'})
            factors_dict['bid_swr'] = swr(datas={'depth5': ob}, params={'side': 'bid'})
        except:
            factors_dict['ask_swr'] = np.zeros(len(ob))
            factors_dict['bid_swr'] = np.zeros(len(ob))
        
        try:
            factors_dict['ask_volume'] = ask_volume(datas={'depth5': ob}, params={})
            factors_dict['bid_volume'] = bid_volume(datas={'depth5': ob}, params={})
        except:
            factors_dict['ask_volume'] = np.zeros(len(ob))
            factors_dict['bid_volume'] = np.zeros(len(ob))
        
        try:
            factors_dict['weakoir'] = weakoir(datas={'depth5': ob}, params={})
        except:
            factors_dict['weakoir'] = np.zeros(len(ob))
        
        try:
            factors_dict['wss'] = wss(datas={'depth5': ob}, params={'n': 5})
        except:
            factors_dict['wss'] = np.zeros(len(ob))
        
        return pd.DataFrame(factors_dict)
    
    def calculate_labels(self, ob, n=600):
        """计算标签"""
        mid_price = (ob['ap1'] + ob['bp1']) / 2
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
            
            # 转换为5档数据
            ob = self.convert_to_depth5(ob_raw)
            tr = self.process_trade_data(tr_raw)
            
            # 检查数据
            if len(ob) == 0 or len(tr) == 0:
                return
            
            # 计算因子和标签
            factors_df = self.calculate_factors(ob, tr)
            labels = self.calculate_labels(ob,n=self.Fre)
            
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
                self.all_factors_df = pd.concat([self.all_factors_df, factors_df], ignore_index=True)
                self.all_labels.extend(labels.tolist())
            
        except Exception:

            return
    
    def save_final_data(self):
        """保存最终的累积数据"""
        # 转换标签为numpy数组
        all_labels_array = np.array(self.all_labels)
        
        # # 保存二进制格式的特征数据
        # feature_values = []
        # for col in self.all_factors_df.columns:
        #     feature_values.extend(self.all_factors_df[col].values)
        # feature_array = np.array(feature_values)
        
        # feature_file_path = os.path.join(self.feature_path, "combined_features")
        # self.save_binary_data(feature_array, feature_file_path)
        
        # # 保存二进制格式的标签数据
        # label_file_path = os.path.join(self.label_path, "combined_labels")
        # self.save_binary_data(all_labels_array, label_file_path)
        
        # 保存H5格式的特征数据
        h5_feature_path = os.path.join(self.feature_path, "combined_features.h5")
        self.all_factors_df.to_hdf(h5_feature_path, key='features', mode='w', format='table', complevel=9)
        
        # 保存H5格式的标签数据
        h5_label_path = os.path.join(self.label_path, "combined_labels.h5")
        pd.DataFrame(all_labels_array, columns=['labels']).to_hdf(h5_label_path, key='labels', mode='w', format='table', complevel=9)
        
        # 保存CSV格式用于检查
        csv_feature_path = os.path.join(self.feature_path, "combined_features.csv")
        self.all_factors_df.to_csv(csv_feature_path, index=False)
        
        csv_label_path = os.path.join(self.label_path, "combined_labels.csv")
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
            print(f"总数据点: {len(self.all_factors_df)}, 因子数: {len(self.all_factors_df.columns)}")
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
    
    # 创建处理器并运行
    processor = H5FactorProcessor(snapshot_dir, trade_dir, output_paths)
    processor.process_all_files()

if __name__ == "__main__":
    main()