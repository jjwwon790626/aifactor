import os
import glob
import struct
import pandas as pd
import numpy as np
from pathlib import Path

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
        
        # 初始化Alpha因子计算器
        self.alpha_calculator = AlphaFactorCalculator()
    
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
        """数据预处理：重命名列并扩展到25档 - 优化版本"""
        print("  数据预处理...")
        
        # ======== 交易数据预处理 ========
        tr = tr.rename(columns={'timestamp': 'ts', 'price': 'p', 'amount': 'v'})
        tr['v'] = np.where(tr['side'] == 'sell', -tr['v'], tr['v'])
        
        # ======== 订单簿数据预处理和扩展 ========
        # 第一步：重命名现有列（保持原有的HFT格式用于HFT因子计算）
        hft_rename_dict = {
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
        }
        
        hft_ob = ob.rename(columns=hft_rename_dict)
        
        # 第二步：创建25档的Alpha格式数据 - 优化版本
        # 检查原始数据中有多少档
        max_available_level = 0
        for i in range(1, 26):  # 检查bp1到bp25
            if f'bp{i}' in hft_ob.columns:
                max_available_level = i
        
        print(f"    原始数据包含 {max_available_level} 档订单簿数据")
        
        # 创建Alpha格式的数据字典，然后一次性构建DataFrame
        alpha_data = {}
        
        # 转换已有的档位数据（从bp1/ap1格式转换为bp0/ap0格式）
        for i in range(max_available_level):
            old_idx = i + 1  # 原数据是bp1, bp2, ..., bp5
            new_idx = i      # 新数据是bp0, bp1, ..., bp4
            
            alpha_data[f'bp{new_idx}'] = hft_ob[f'bp{old_idx}'].values
            alpha_data[f'ap{new_idx}'] = hft_ob[f'ap{old_idx}'].values
            alpha_data[f'bv{new_idx}'] = hft_ob[f'bv{old_idx}'].values
            alpha_data[f'av{new_idx}'] = hft_ob[f'av{old_idx}'].values
        
        # 第三步：如果原始数据不足25档，需要填充缺失的档位
        if max_available_level < 25:
            last_available_idx = max_available_level - 1  # 最后一个可用档位的索引（转换后的索引）
            
            print(f"    需要填充第 {max_available_level} 档到第 25 档的数据")
            
            # 获取最后可用档位的数据用于填充
            if last_available_idx >= 0:
                last_bp = alpha_data[f'bp{last_available_idx}']
                last_ap = alpha_data[f'ap{last_available_idx}']
                last_bv = alpha_data[f'bv{last_available_idx}']
                last_av = alpha_data[f'av{last_available_idx}']
                
                for i in range(max_available_level, 25):  # 填充缺失档位
                    # 价格：在最后可用价格基础上按买卖方向递减/递增
                    price_decay_factor = 1 - 0.0005 * (i - last_available_idx)  # 买入价格递减
                    price_growth_factor = 1 + 0.0005 * (i - last_available_idx)  # 卖出价格递增
                    
                    alpha_data[f'bp{i}'] = last_bp * price_decay_factor
                    alpha_data[f'ap{i}'] = last_ap * price_growth_factor
                    
                    # 成交量：使用指数衰减函数，距离越远成交量越小
                    volume_decay_factor = 0.8 ** (i - last_available_idx)
                    alpha_data[f'bv{i}'] = last_bv * volume_decay_factor
                    alpha_data[f'av{i}'] = last_av * volume_decay_factor
            else:
                # 如果连第一档都没有，用默认值填充
                print("    警告：原始数据中没有任何档位信息，使用默认值填充")
                for i in range(25):
                    alpha_data[f'bp{i}'] = np.full(len(hft_ob), 100.0)  # 默认买入价格
                    alpha_data[f'ap{i}'] = np.full(len(hft_ob), 101.0)  # 默认卖出价格
                    alpha_data[f'bv{i}'] = np.full(len(hft_ob), 1.0)    # 默认成交量
                    alpha_data[f'av{i}'] = np.full(len(hft_ob), 1.0)    # 默认成交量
        
        # 添加时间戳
        if 'ts' in hft_ob.columns:
            alpha_data['ts'] = hft_ob['ts'].values
        
        # 一次性创建DataFrame，避免碎片化
        alpha_ob = pd.DataFrame(alpha_data, index=hft_ob.index)
        
        # 第四步：数据清理和验证
        # 使用新的fillna语法
        alpha_ob = alpha_ob.ffill().bfill().fillna(0)
        
        # 确保价格和成交量都是正数
        for i in range(25):
            alpha_ob[f'bp{i}'] = alpha_ob[f'bp{i}'].abs()
            alpha_ob[f'ap{i}'] = alpha_ob[f'ap{i}'].abs()
            alpha_ob[f'bv{i}'] = alpha_ob[f'bv{i}'].abs()
            alpha_ob[f'av{i}'] = alpha_ob[f'av{i}'].abs()
            
            # 确保买入价格 <= 卖出价格的逻辑顺序
            if i > 0:
                # 买入价格应该递减
                alpha_ob[f'bp{i}'] = np.minimum(alpha_ob[f'bp{i}'], alpha_ob[f'bp{i-1}'])
                # 卖出价格应该递增
                alpha_ob[f'ap{i}'] = np.maximum(alpha_ob[f'ap{i}'], alpha_ob[f'ap{i-1}'])
        
        # 验证数据完整性
        expected_cols = [f'{prefix}{i}' for prefix in ['bp', 'ap', 'bv', 'av'] for i in range(25)]
        missing_cols = set(expected_cols) - set(alpha_ob.columns)
        if missing_cols:
            print(f"    警告：仍有缺失列: {missing_cols}")
            # 用零值填充缺失列
            for col in missing_cols:
                alpha_ob[col] = 0.0
        
        print(f"    数据预处理完成:")
        print(f"    - HFT格式: {len(hft_ob.columns)} 列，{len(hft_ob)} 行")
        print(f"    - Alpha格式: {len(alpha_ob.columns)} 列，{len(alpha_ob)} 行")
        
        return hft_ob, alpha_ob, tr
    
    def calculate_factors(self, hft_ob, alpha_ob, tr):
        """计算所有因子 - 优化版本"""
        print("  计算因子...")
        
        # 存储HFT因子数据的字典
        hft_factors_data = {}
        
        # 计算HFT因子（使用HFT格式的数据）
        print("    计算HFT因子...")
        try:
            # 到达率因子
            hft_factors_data['arrive_rate'] = arrive_rate(
                datas={'depth5': hft_ob, 'trade': tr},
                params={'n': 600}
            )
            
            # 公平价差因子
            hft_factors_data['cofi'] = cofi(datas={'depth5': hft_ob}, params={})
            
            # 深度账簿年龄因子
            hft_factors_data['depth_bid_age'] = depth_bid_age(datas={'depth5': hft_ob}, params={'n': self.Fre})
            hft_factors_data['depth_ask_age'] = depth_ask_age(datas={'depth5': hft_ob}, params={'n': self.Fre})
            
            # 深度账簿变化因子
            hft_factors_data['depth_bid_change'] = depth_bid_change(datas={'depth5': hft_ob}, params={'n': self.Fre})
            hft_factors_data['depth_ask_change'] = depth_ask_change(datas={'depth5': hft_ob}, params={'n': self.Fre})
            
            # 公平价差因子
            hft_factors_data['fair_spread'] = fair_spread(datas={'depth5': hft_ob}, params={})
            
            # 大跳因子
            hft_factors_data['large_jump_up'] = large_jump(
                datas={'depth5': hft_ob},
                params={'n': self.Fre, 'direct': 'up', 'jump_ratio': 0.001}
            )
            hft_factors_data['large_jump_down'] = large_jump(
                datas={'depth5': hft_ob},
                params={'n': 100, 'direct': 'down', 'jump_ratio': 0.001}
            )
            
            # LLT平滑值
            hft_factors_data['llt'] = llt(datas={"depth5": hft_ob}, params={"n": self.Fre})
            
            # 订单失衡比率
            hft_factors_data['oir'] = oir(datas={'depth5': hft_ob}, params={})
            
            # 订单流因子
            hft_factors_data['bid_order_flow'] = oflow(datas={'depth5': hft_ob}, params={'side': 'bid', 'bend_ratio': 4})
            hft_factors_data['ask_order_flow'] = oflow(datas={'depth5': hft_ob}, params={'side': 'ask', 'bend_ratio': 4})
            
            # 价格距离因子
            hft_factors_data['ask_price_distance'] = price_distance(datas={'depth5': hft_ob}, params={'n': self.Fre, 'side': 'ask'})
            hft_factors_data['bid_price_distance'] = price_distance(datas={'depth5': hft_ob}, params={'n': self.Fre, 'side': 'bid'})
            
            # 价格冲击因子
            hft_factors_data['price_impact'] = price_impact(datas={'depth5': hft_ob}, params={'n': 5})
            
            # 滞后因子
            hft_factors_data['ask_lag'] = ask_lag(datas={'depth5': hft_ob}, params={'n': self.Fre})
            hft_factors_data['bid_lag'] = bid_lag(datas={'depth5': hft_ob}, params={'n': self.Fre})
            
            # SWR因子
            hft_factors_data['ask_swr'] = swr(datas={'depth5': hft_ob}, params={'side': 'ask'})
            hft_factors_data['bid_swr'] = swr(datas={'depth5': hft_ob}, params={'side': 'bid'})
            
            # tick_vpin因子
            hft_factors_data['tick_vpin'] = tick_vpin(datas={'depth5': hft_ob, 'trade': tr}, params={'n': 600})
            
            # volume_at_same_price因子
            hft_factors_data['ask_volume_at_same_price'] = volume_at_same_price(datas={'depth5': hft_ob, 'trade': tr}, params={'n': 600})
            
            # 交易量因子
            hft_factors_data['ask_volume'] = ask_volume(datas={'depth5': hft_ob}, params={})
            hft_factors_data['bid_volume'] = bid_volume(datas={'depth5': hft_ob}, params={})
            
            # 弱订单失衡比率
            hft_factors_data['weakoir'] = weakoir(datas={'depth5': hft_ob}, params={})
            
            # WSS因子
            hft_factors_data['wss'] = wss(datas={'depth5': hft_ob}, params={'n': 5})
            
            # 一次性创建HFT因子DataFrame
            hft_factors_df = pd.DataFrame(hft_factors_data, index=hft_ob.index)
            
            print(f"    HFT因子计算完成，共 {len(hft_factors_df.columns)} 个因子")
            
        except Exception as e:
            print(f"    HFT因子计算出错: {str(e)}")
            # 如果HFT因子计算失败，创建空的DataFrame
            hft_factors_df = pd.DataFrame(index=hft_ob.index)
        
        # 计算Alpha因子（使用预处理好的25档Alpha格式数据）
        print("    计算Alpha因子...")
        try:
            # 直接使用预处理好的alpha_ob数据，无需再次转换
            alpha_factors_df = self.alpha_calculator.calculate_all_factors(alpha_ob)
            
            print(f"    Alpha因子计算完成，共 {len(alpha_factors_df.columns)} 个因子")
            
        except Exception as e:
            print(f"    Alpha因子计算出错: {str(e)}")
            # 如果Alpha因子计算失败，创建空的DataFrame
            alpha_factors_df = pd.DataFrame(index=alpha_ob.index)
        
        # 合并所有因子 - 优化版本
        print("    合并因子...")
        
        # 确保所有DataFrame的索引一致
        min_length = min(len(hft_factors_df), len(alpha_factors_df), len(hft_ob))
        
        if min_length > 0:
            # 截取到相同长度
            hft_factors_df = hft_factors_df.iloc[:min_length]
            alpha_factors_df = alpha_factors_df.iloc[:min_length]
            
            # 重置索引以确保合并正确
            hft_factors_df.reset_index(drop=True, inplace=True)
            alpha_factors_df.reset_index(drop=True, inplace=True)
            
            # 使用pd.concat一次性合并HFT因子和Alpha因子
            combined_factors_df = pd.concat([hft_factors_df, alpha_factors_df], axis=1)
        else:
            print("    警告：数据长度为0，创建空的因子DataFrame")
            combined_factors_df = pd.DataFrame()
        
        print(f"    因子合并完成，总共 {len(combined_factors_df.columns)} 个因子")
        
        return combined_factors_df
    
    def calculate_labels(self, hft_ob):
        """计算标签"""
        mid_price = (hft_ob['ap1'] + hft_ob['bp1']) / 2
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
        
        # 数据预处理（现在返回三个对象：HFT格式、Alpha格式、交易数据）
        hft_ob, alpha_ob, tr = self.preprocess_data(ob, tr)
        
        # 计算因子
        factors_df = self.calculate_factors(hft_ob, alpha_ob, tr)
        
        # 如果因子计算失败，跳过这个文件
        if factors_df.empty:
            print(f"  跳过文件 {base_name}：因子计算失败")
            return
        
        # 计算标签（使用HFT格式数据）
        labels = self.calculate_labels(hft_ob)
        
        # 确保数据长度一致
        min_length = min(len(factors_df), len(labels))
        factors_df = factors_df.iloc[:min_length].fillna(0)
        labels = labels[:min_length]
        
        # 累积数据 - 优化版本
        if self.all_factors_df.empty:
            self.all_factors_df = factors_df.copy()
            self.all_labels = labels.tolist()
            self.save_factor_names(factors_df.columns)
        else:
            # 确保列名一致
            if set(factors_df.columns) == set(self.all_factors_df.columns):
                # 使用pd.concat进行高效合并
                self.all_factors_df = pd.concat([self.all_factors_df, factors_df], ignore_index=True)
                self.all_labels.extend(labels.tolist())
            else:
                print(f"  警告：文件 {base_name} 的因子列名与之前不一致，跳过")
                return
        
        print(f"  处理完成，当前累积数据长度: {len(self.all_factors_df)}")
    
    def save_final_data(self):
        """保存最终的累积数据"""
        print("保存最终数据...")
        
        # 转换标签为numpy数组
        all_labels_array = np.array(self.all_labels)
        
        # 保存特征数据 - 优化版本
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