import pandas as pd
import numpy as np
import gzip
import os
from datetime import datetime, timedelta
import warnings

# 设置pandas选项以避免FutureWarning
pd.set_option('future.no_silent_downcasting', True)
warnings.filterwarnings('ignore', category=FutureWarning)

def read_compressed_csv(file_path):
    """读取压缩的CSV文件"""
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            df = pd.read_csv(f)
        return df
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {e}")
        return None

def calculate_ma(prices, period):
    """计算移动平均线"""
    return prices.rolling(window=period).mean()

def process_snapshot_to_kline(df, timeframe='15min'):
    """将快照数据转换为15分钟K线数据"""
    # 转换时间戳为datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='us')
    df = df.set_index('datetime')
    
    # 检查列名，处理可能的列名变化
    ask_price_col = None
    bid_price_col = None
    
    # 寻找asks和bids价格列
    for col in df.columns:
        if 'asks[0].price' in col or 'asks_0_price' in col:
            ask_price_col = col
        elif 'bids[0].price' in col or 'bids_0_price' in col:
            bid_price_col = col
    
    # 如果没找到标准列名，尝试其他可能的列名
    if ask_price_col is None:
        potential_ask_cols = [col for col in df.columns if 'ask' in col.lower() and 'price' in col.lower()]
        if potential_ask_cols:
            ask_price_col = potential_ask_cols[0]
    
    if bid_price_col is None:
        potential_bid_cols = [col for col in df.columns if 'bid' in col.lower() and 'price' in col.lower()]
        if potential_bid_cols:
            bid_price_col = potential_bid_cols[0]
    
    if ask_price_col is None or bid_price_col is None:
        print("可用的列名:")
        print(df.columns.tolist())
        raise ValueError(f"无法找到ask/bid价格列。Ask列: {ask_price_col}, Bid列: {bid_price_col}")
    
    # 使用mid price作为价格 (asks[0].price + bids[0].price) / 2
    df['mid_price'] = (df[ask_price_col] + df[bid_price_col]) / 2
    
    # 重采样为15分钟K线
    kline = df['mid_price'].resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    return kline

def find_ma_trend_periods(kline_df):
    """找到三均线多头/空头的完整周期"""
    # 计算三条移动平均线 (5, 10, 20)
    ma_5 = calculate_ma(kline_df['close'], 5)
    ma_10 = calculate_ma(kline_df['close'], 10)
    ma_20 = calculate_ma(kline_df['close'], 20)
    
    # 检测多头排列：MA5 > MA10 > MA20
    bull_alignment = ((ma_5 > ma_10) & (ma_10 > ma_20)).fillna(False)
    
    # 检测空头排列：MA5 < MA10 < MA20
    bear_alignment = ((ma_5 < ma_10) & (ma_10 < ma_20)).fillna(False)
    
    trend_periods = []
    
    # 寻找多头排列周期
    in_bull_trend = False
    bull_start = None
    
    for i in range(len(bull_alignment)):
        if bull_alignment.iloc[i] and not in_bull_trend:
            in_bull_trend = True
            bull_start = i
        elif not bull_alignment.iloc[i] and in_bull_trend:
            in_bull_trend = False
            bull_end = i - 1
            
            if bull_end - bull_start >= 0:  # 至少1根K线
                trend_periods.append({
                    'trend_type': 'bull',
                    'trend_category': '三均线多头',
                    'start_time': bull_alignment.index[bull_start],
                    'end_time': bull_alignment.index[bull_end] + pd.Timedelta(minutes=15),  # 加15分钟包含最后一根K线
                    'duration_bars': bull_end - bull_start + 1
                })
    
    # 处理最后一个多头周期
    if in_bull_trend and bull_start is not None:
        bull_end = len(bull_alignment) - 1
        if bull_end - bull_start >= 0:
            trend_periods.append({
                'trend_type': 'bull',
                'trend_category': '三均线多头',
                'start_time': bull_alignment.index[bull_start],
                'end_time': bull_alignment.index[bull_end] + pd.Timedelta(minutes=15),
                'duration_bars': bull_end - bull_start + 1
            })
    
    # 寻找空头排列周期
    in_bear_trend = False
    bear_start = None
    
    for i in range(len(bear_alignment)):
        if bear_alignment.iloc[i] and not in_bear_trend:
            in_bear_trend = True
            bear_start = i
        elif not bear_alignment.iloc[i] and in_bear_trend:
            in_bear_trend = False
            bear_end = i - 1
            
            if bear_end - bear_start >= 0:  # 至少1根K线
                trend_periods.append({
                    'trend_type': 'bear',
                    'trend_category': '三均线空头',
                    'start_time': bear_alignment.index[bear_start],
                    'end_time': bear_alignment.index[bear_end] + pd.Timedelta(minutes=15),
                    'duration_bars': bear_end - bear_start + 1
                })
    
    # 处理最后一个空头周期
    if in_bear_trend and bear_start is not None:
        bear_end = len(bear_alignment) - 1
        if bear_end - bear_start >= 0:
            trend_periods.append({
                'trend_type': 'bear',
                'trend_category': '三均线空头',
                'start_time': bear_alignment.index[bear_start],
                'end_time': bear_alignment.index[bear_end] + pd.Timedelta(minutes=15),
                'duration_bars': bear_end - bear_start + 1
            })
    
    return trend_periods

def filter_data_by_timerange(df, start_time, end_time):
    """根据时间范围筛选数据"""
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='us')
    mask = (df['datetime'] >= start_time) & (df['datetime'] < end_time)
    return df[mask]

def main():
    # 配置路径
    raw_data_path = "/Users/wook/Downloads/因子挖掘/raw_data"
    output_base_path = "/Users/wook/Downloads/因子挖掘/Sample data/raw/BTC分段数据"
    
    # 创建输出文件夹
    categories = ['三均线多头', '三均线空头']
    category_paths = {}
    
    for category in categories:
        book_path = os.path.join(output_base_path, category, "book25_snapshot")
        trade_path = os.path.join(output_base_path, category, "trade")
        os.makedirs(book_path, exist_ok=True)
        os.makedirs(trade_path, exist_ok=True)
        category_paths[category] = {
            'book': book_path,
            'trade': trade_path
        }
    
    # 日期范围
    start_date = "20250502"
    end_date = "20250609"
    
    # 生成日期列表
    date_range = []
    current = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    
    while current <= end:
        date_range.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    
    all_trend_periods = []
    category_counters = {category: 0 for category in categories}
    
    # 处理每个日期的数据
    for date in date_range:
        print(f"处理日期: {date}")
        
        # 文件路径
        snapshot_file = os.path.join(raw_data_path, f"BTCUSDT_book_snapshot_25_{date}.csv.gz")
        trade_file = os.path.join(raw_data_path, f"BTCUSDT_trades_{date}.csv.gz")
        
        if not os.path.exists(snapshot_file) or not os.path.exists(trade_file):
            continue
        
        # 读取快照数据并转换为15分钟K线
        snapshot_df = read_compressed_csv(snapshot_file)
        if snapshot_df is None:
            continue
            
        try:
            kline_df = process_snapshot_to_kline(snapshot_df, '15min')
            if len(kline_df) < 30:
                continue
        except Exception as e:
            print(f"处理K线数据失败 {date}: {e}")
            continue
        
        # 找到趋势周期
        trend_periods = find_ma_trend_periods(kline_df)
        if not trend_periods:
            continue
        
        # 读取交易数据
        trade_df = read_compressed_csv(trade_file)
        if trade_df is None:
            continue
        
        # 保存每个趋势周期的完整数据
        for period in trend_periods:
            category = period['trend_category']
            category_counters[category] += 1
            
            # 生成文件名
            segment_name = (f"{category}_{date}_{category_counters[category]:03d}_"
                          f"{period['start_time'].strftime('%H%M')}-{period['end_time'].strftime('%H%M')}_"
                          f"{period['duration_bars']}bars")
            
            # 筛选和保存快照数据（整个趋势周期的数据）
            filtered_snapshot = filter_data_by_timerange(
                snapshot_df, period['start_time'], period['end_time']
            )
            
            if len(filtered_snapshot) > 0:
                snapshot_output_file = os.path.join(category_paths[category]['book'], f"{segment_name}_snapshot.h5")
                filtered_snapshot.to_hdf(snapshot_output_file, key='snapshot', mode='w', format='table', complevel=9)
                
                # 筛选和保存交易数据
                filtered_trade = filter_data_by_timerange(
                    trade_df, period['start_time'], period['end_time']
                )
                
                if len(filtered_trade) > 0:
                    trade_output_file = os.path.join(category_paths[category]['trade'], f"{segment_name}_trade.h5")
                    filtered_trade.to_hdf(trade_output_file, key='trade', mode='w', format='table', complevel=9)
                
                all_trend_periods.append({
                    'date': date,
                    'segment_name': segment_name,
                    'trend_category': category,
                    'start_time': period['start_time'],
                    'end_time': period['end_time'],
                    'duration_bars': period['duration_bars']
                })
    
    # 保存汇总信息
    if all_trend_periods:
        summary_df = pd.DataFrame(all_trend_periods)
        summary_file = os.path.join(output_base_path, "ma_trend_periods_summary.h5")
        summary_df.to_hdf(summary_file, key='summary', mode='w', format='table', complevel=9)
        print(f"总共处理 {len(all_trend_periods)} 个趋势周期，汇总保存到: {summary_file}")

if __name__ == "__main__":
    main()