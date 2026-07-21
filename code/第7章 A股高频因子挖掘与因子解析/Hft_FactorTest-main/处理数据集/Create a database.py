import os
import pandas as pd
import shutil
import random

def get_h5_file_pairs(folder_path):
    """获取文件夹中成对的book和trade H5文件"""
    book_folder = os.path.join(folder_path, "book25_snapshot")
    trade_folder = os.path.join(folder_path, "trade")
    
    if not os.path.exists(book_folder) or not os.path.exists(trade_folder):
        return []
    
    book_files = [f for f in os.listdir(book_folder) if f.endswith('_snapshot.h5')]
    trade_files = [f for f in os.listdir(trade_folder) if f.endswith('_trade.h5')]
    
    pairs = []
    for book_file in book_files:
        base_name = book_file.replace('_snapshot.h5', '')
        trade_file = base_name + '_trade.h5'
        
        if trade_file in trade_files:
            pairs.append({
                'base_name': base_name,
                'book_path': os.path.join(book_folder, book_file),
                'trade_path': os.path.join(trade_folder, trade_file),
                'category': '三均线多头' if '多头' in book_file else '三均线空头'
            })
    
    return pairs

def collect_all_data_pairs(base_path):
    """收集所有数据对并分类"""
    categories = ['三均线多头', '三均线空头']
    long_pairs = []
    short_pairs = []
    
    for category in categories:
        category_path = os.path.join(base_path, category)
        if not os.path.exists(category_path):
            continue
        
        pairs = get_h5_file_pairs(category_path)
        print(f"{category}: {len(pairs)} 对文件")
        
        if '多头' in category:
            long_pairs.extend(pairs)
        else:
            short_pairs.extend(pairs)
    
    return long_pairs, short_pairs

def balance_and_copy_data(long_pairs, short_pairs, output_book_path, output_trade_path):
    """平衡数据并复制文件"""
    print(f"做多数据: {len(long_pairs)} 对")
    print(f"做空数据: {len(short_pairs)} 对")
    
    min_count = min(len(long_pairs), len(short_pairs))
    if min_count == 0:
        print("数据为空!")
        return 0, 0
    
    print(f"平衡后每类数据: {min_count} 对")
    
    random.seed(42)
    balanced_long = random.sample(long_pairs, min_count)
    balanced_short = random.sample(short_pairs, min_count)
    
    # 复制做多数据
    for i, pair in enumerate(balanced_long):
        base_name = f"LONG_{i+1:04d}_MA多头"
        shutil.copy2(pair['book_path'], os.path.join(output_book_path, f"{base_name}_snapshot.h5"))
        shutil.copy2(pair['trade_path'], os.path.join(output_trade_path, f"{base_name}_trade.h5"))
    
    # 复制做空数据
    for i, pair in enumerate(balanced_short):
        base_name = f"SHORT_{i+1:04d}_MA空头"
        shutil.copy2(pair['book_path'], os.path.join(output_book_path, f"{base_name}_snapshot.h5"))
        shutil.copy2(pair['trade_path'], os.path.join(output_trade_path, f"{base_name}_trade.h5"))
    
    return len(balanced_long), len(balanced_short)

def main():
    # 配置路径
    input_base_path = "/Users/wook/Downloads/因子挖掘/Sample data/raw/BTC分段数据"
    output_base_path = "/Users/wook/Downloads/因子挖掘/Sample data/raw/筛选BTC分段数据"
    
    # 创建输出文件夹
    output_book_path = os.path.join(output_base_path, "book25_snapshot")
    output_trade_path = os.path.join(output_base_path, "trade")
    os.makedirs(output_book_path, exist_ok=True)
    os.makedirs(output_trade_path, exist_ok=True)
    
    # 收集和处理数据
    long_pairs, short_pairs = collect_all_data_pairs(input_base_path)
    
    if not long_pairs and not short_pairs:
        print("没有找到任何H5数据对!")
        return
    
    # 平衡数据并复制文件
    long_copied, short_copied = balance_and_copy_data(long_pairs, short_pairs, output_book_path, output_trade_path)
    
    print(f"\n处理完成!")
    print(f"做多数据: {long_copied} 对")
    print(f"做空数据: {short_copied} 对")
    print(f"总计: {long_copied + short_copied} 对")
    
    # 验证结果
    book_files = len([f for f in os.listdir(output_book_path) if f.endswith('.h5')])
    trade_files = len([f for f in os.listdir(output_trade_path) if f.endswith('.h5')])
    
    if book_files == trade_files == (long_copied + short_copied):
        print("✅ 验证通过!")
    else:
        print("❌ 文件数量不匹配!")

if __name__ == "__main__":
    main()