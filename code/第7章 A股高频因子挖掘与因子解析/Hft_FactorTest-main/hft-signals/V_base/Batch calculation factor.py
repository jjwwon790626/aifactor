import os
import glob
import struct
import pandas as pd
import numpy as np
from pathlib import Path

import sys
# 添加 hft 库路径
hft_signals_path = '/Users/wook/Downloads/FactorTest1/hft-signals'
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

class FactorProcessor:
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
    
    def get_file_pairs(self):
        """获取配对的快照和交易文件"""
        snapshot_files = glob.glob(os.path.join(self.snapshot_dir, "*_snapshot.csv"))
        file_pairs = []
        
        for snapshot_file in snapshot_files:
            # 提取基础文件名（去掉_snapshot.csv）
            base_name = os.path.basename(snapshot_file).replace('_snapshot.csv', '')
            trade_file = os.path.join(self.trade_dir, base_name + '_trade.csv')
            
            if os.path.exists(trade_file):
                file_pairs.append((snapshot_file, trade_file, base_name))
            else:
                print(f"警告: 找不到对应的交易文件: {trade_file}")
        
        return sorted(file_pairs)
    
    def preprocess_data(self, ob, tr):
        """数据预处理：重命名列"""
        # 订单簿数据重命名
        ob = ob.rename(columns={
            'timestamp': 'ts',
            'bids[0].price': 'bp1', 'bids[0].amount': 'bv1',
            'asks[0].price': 'ap1', 'asks[0].amount': 'av1',
            'bids[1].price': 'bp2', 'bids[1].amount': 'bv2',
            'asks[1].price': 'ap2', 'asks[1].amount': 'av2',
            'bids[2].price': 'bp3', 'bids[2].amount': 'bv3',
            'asks[2].price': 'ap3', 'asks[2].amount': 'av3',
            'bids[3].price': 'bp4', 'bids[3].amount': 'bv4',
            'asks[3].price': 'ap4', 'asks[3].amount': 'av4',
            'bids[4].price': 'bp5', 'bids[4].amount': 'bv5',
            'asks[4].price': 'ap5', 'asks[4].amount': 'av5'
        })
        
        # 交易数据重命名和处理
        tr = tr.rename(columns={'timestamp': 'ts', 'price': 'p', 'amount': 'v'})
        tr['v'] = np.where(tr['side'] == 'sell', -tr['v'], tr['v'])
        
        return ob, tr
    
    def calculate_factors(self, ob, tr):
        """计算所有因子"""
        print("  计算因子...")
        
        # 到达率因子
        arrive_rate_factor = arrive_rate(
            datas={'depth5': ob, 'trade': tr},
            params={'n': 600}
        )
        
        # 公平价差因子
        cofi_factor = cofi(datas={'depth5': ob}, params={})
        
        # 深度账簿年龄因子
        depth_bid_age_factor = depth_bid_age(datas={'depth5': ob}, params={'n': self.Fre})
        depth_ask_age_factor = depth_ask_age(datas={'depth5': ob}, params={'n': self.Fre})
        
        # 深度账簿变化因子
        depth_bid_change_factor = depth_bid_change(datas={'depth5': ob}, params={'n': self.Fre})
        depth_ask_change_factor = depth_ask_change(datas={'depth5': ob}, params={'n': self.Fre})
        
        # 公平价差因子
        fair_spread_factor = fair_spread(datas={'depth5': ob}, params={})
        
        # 大跳因子
        large_jump_up_factor = large_jump(
            datas={'depth5': ob},
            params={'n': self.Fre, 'direct': 'up', 'jump_ratio': 0.001}
        )
        large_jump_down_factor = large_jump(
            datas={'depth5': ob},
            params={'n': 100, 'direct': 'down', 'jump_ratio': 0.001}
        )
        
        # LLT平滑值
        llt_factor = llt(datas={"depth5": ob}, params={"n": self.Fre})
        
        # 订单失衡比率
        oir_factor = oir(datas={'depth5': ob}, params={})
        
        # 订单流因子
        bid_order_flow_factor = oflow(datas={'depth5': ob}, params={'side': 'bid', 'bend_ratio': 4})
        ask_order_flow_factor = oflow(datas={'depth5': ob}, params={'side': 'ask', 'bend_ratio': 4})
        
        # 价格距离因子
        ask_price_distance_factor = price_distance(datas={'depth5': ob}, params={'n': self.Fre, 'side': 'ask'})
        bid_price_distance_factor = price_distance(datas={'depth5': ob}, params={'n': self.Fre, 'side': 'bid'})
        
        # 价格冲击因子
        price_impact_factor = price_impact(datas={'depth5': ob}, params={'n': 5})
        
        # 滞后因子
        ask_lag_factor = ask_lag(datas={'depth5': ob}, params={'n': self.Fre})
        bid_lag_factor = bid_lag(datas={'depth5': ob}, params={'n': self.Fre})
        
        # SWR因子
        ask_swr_factor = swr(datas={'depth5': ob}, params={'side': 'ask'})
        bid_swr_factor = swr(datas={'depth5': ob}, params={'side': 'bid'})
        
        # tick_vpin因子
        tick_vpin_factor = tick_vpin(datas={'depth5': ob, 'trade': tr}, params={'n': 600})
        
        # volume_at_same_price因子
        volume_at_same_price_factor = volume_at_same_price(datas={'depth5': ob, 'trade': tr}, params={'n': 600})
        
        # 交易量因子
        ask_volume_factor = ask_volume(datas={'depth5': ob}, params={})
        bid_volume_factor = bid_volume(datas={'depth5': ob}, params={})
        
        # 弱订单失衡比率
        weakoir_factor = weakoir(datas={'depth5': ob}, params={})
        
        # WSS因子
        wss_factor = wss(datas={'depth5': ob}, params={'n': 5})
        
        # 合并所有因子
        factors_df = pd.DataFrame({
            'arrive_rate': arrive_rate_factor,
            'cofi': cofi_factor,
            'depth_bid_age': depth_bid_age_factor,
            'depth_ask_age': depth_ask_age_factor,
            'depth_bid_change': depth_bid_change_factor,
            'depth_ask_change': depth_ask_change_factor,
            'fair_spread': fair_spread_factor,
            'large_jump_up': large_jump_up_factor,
            'large_jump_down': large_jump_down_factor,
            'llt': llt_factor,
            'oir': oir_factor,
            'bid_order_flow': bid_order_flow_factor,
            'ask_order_flow': ask_order_flow_factor,
            'ask_price_distance': ask_price_distance_factor,
            'bid_price_distance': bid_price_distance_factor,
            'price_impact': price_impact_factor,
            'ask_lag': ask_lag_factor,
            'bid_lag': bid_lag_factor,
            'ask_swr': ask_swr_factor,
            'bid_swr': bid_swr_factor,
            'tick_vpin': tick_vpin_factor,
            'ask_volume_at_same_price': volume_at_same_price_factor,
            'ask_volume': ask_volume_factor,
            'bid_volume': bid_volume_factor,
            'weakoir': weakoir_factor,
            'wss': wss_factor
        })
        
        return factors_df
    
    def calculate_labels(self, ob):
        """计算标签"""
        mid_price = (ob['ap1'] + ob['bp1']) / 2
        # 直接计算未来600期的收益率
        labels = mid_price.pct_change(600).shift(-600).fillna(0)
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
            print(f"因子名称已保存到: {factor_names_path}")
            self.factor_names_saved = True
    
    def process_single_file(self, snapshot_file, trade_file, base_name):
        """处理单个文件对"""
        print(f"处理文件: {base_name}")
        
        # 读取数据
        print("  读取数据...")
        ob = pd.read_csv(snapshot_file)
        tr = pd.read_csv(trade_file)
        
        # 数据预处理
        ob, tr = self.preprocess_data(ob, tr)
        
        # 计算因子
        factors_df = self.calculate_factors(ob, tr)
        
        # 计算标签
        labels = self.calculate_labels(ob)
        
        # 确保数据长度一致
        min_length = min(len(factors_df), len(labels))
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
        
        print(f"  处理完成，当前累积数据长度: {len(self.all_factors_df)}")
    
    def save_final_data(self):
        """保存最终的累积数据"""
        print("保存最终数据...")
        
        # 转换标签为numpy数组
        all_labels_array = np.array(self.all_labels)
        
        # 保存特征数据
        feature_values = []
        for col in self.all_factors_df.columns:
            feature_values.extend(self.all_factors_df[col].values)
        feature_array = np.array(feature_values)
        
        feature_file_path = os.path.join(self.feature_path, "combined_features")
        self.save_binary_data(feature_array, feature_file_path)
        print(f"特征数据已保存到: {feature_file_path}")
        
        # 保存标签数据
        label_file_path = os.path.join(self.label_path, "combined_labels")
        self.save_binary_data(all_labels_array, label_file_path)
        print(f"标签数据已保存到: {label_file_path}")
        
        # 打印统计信息
        print(f"\n最终数据统计:")
        print(f"- 时间点数量: {len(self.all_factors_df)}")
        print(f"- 特征数量: {len(self.all_factors_df.columns)}")
        print(f"- 特征数据文件: {feature_file_path}")
        print(f"- 标签数据文件: {label_file_path}")
    
    def process_all_files(self):
        """处理所有文件"""
        file_pairs = self.get_file_pairs()
        
        if not file_pairs:
            print("未找到任何配对的数据文件！")
            return
        
        print(f"找到 {len(file_pairs)} 对文件")
        
        for i, (snapshot_file, trade_file, base_name) in enumerate(file_pairs, 1):
            print(f"\n[{i}/{len(file_pairs)}] 处理文件对...")
            try:
                self.process_single_file(snapshot_file, trade_file, base_name)
            except Exception as e:
                print(f"处理文件 {base_name} 时出错: {str(e)}")
                continue
        
        # 保存最终数据
        if not self.all_factors_df.empty:
            self.save_final_data()
        else:
            print("没有成功处理任何数据文件！")

def main():
    # 设置路径
    snapshot_dir = "/Users/wook/Downloads/因子挖掘/Sample data/raw/筛选BTC分段数据/book25_snapshot_1s"
    trade_dir = "/Users/wook/Downloads/因子挖掘/Sample data/raw/筛选BTC分段数据/trade"
    
    output_paths = {
        'feature_path': "/Users/wook/Downloads/FactorTest1/feature_test_data/feature",
        'label_path': "/Users/wook/Downloads/FactorTest1/feature_test_data/label",
        'factor_name_path': "/Users/wook/Downloads/FactorTest1/feature_error_test"
    }
    
    # 创建处理器并运行
    processor = FactorProcessor(snapshot_dir, trade_dir, output_paths)
    processor.process_all_files()
    
    print("\n批量处理完成！")

if __name__ == "__main__":
    main()