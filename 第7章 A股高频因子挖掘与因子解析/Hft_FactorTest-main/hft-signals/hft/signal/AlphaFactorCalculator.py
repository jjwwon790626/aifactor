import numpy as np
import pandas as pd
from typing import Dict, Callable


class AlphaFactorCalculator:
    """
    Alpha因子计算器类
    
    用于计算160个alpha因子，输入订单簿数据，返回包含所有因子的DataFrame
    
    使用方法:
    calculator = AlphaFactorCalculator()
    result_df = calculator.calculate_all_factors(df)
    """
    
    def __init__(self):
        """初始化因子计算器"""
        self.factor_methods = self._get_factor_methods()
    
    def _get_factor_methods(self) -> Dict[str, Callable]:
        """获取所有因子计算方法的字典"""
        factor_methods = {}
        
        # 通过反射获取所有factor_xxx方法
        for attr_name in dir(self):
            if attr_name.startswith('factor_') and callable(getattr(self, attr_name)):
                factor_methods[attr_name] = getattr(self, attr_name)
        
        return factor_methods
    
    def calculate_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有因子 - 优化版本，避免DataFrame碎片化
        
        Args:
            df: 包含订单簿数据的DataFrame，需要包含bp0-bp24, ap0-ap24, bv0-bv24, av0-av24列
            
        Returns:
            包含所有因子值的DataFrame，列名为factor_001到factor_160
        """
        # 存储所有因子数据的字典
        factors_data = {}
        
        # 计算所有因子
        for factor_name in sorted(self.factor_methods.keys()):
            try:
                factor_values = self.factor_methods[factor_name](df)
                factors_data[factor_name] = factor_values
            except Exception as e:
                print(f"计算 {factor_name} 时出错: {str(e)}")
                factors_data[factor_name] = np.full(len(df), np.nan)
        
        # 一次性创建DataFrame，避免碎片化
        result = pd.DataFrame(factors_data, index=df.index)
        
        return result
        
    def calculate_single_factor(self, df: pd.DataFrame, factor_name: str) -> np.ndarray:
        """
        计算单个因子
        
        Args:
            df: 包含订单簿数据的DataFrame
            factor_name: 因子名称，如 'factor_001'
            
        Returns:
            因子值数组
        """
        if factor_name not in self.factor_methods:
            raise ValueError(f"未找到因子: {factor_name}")
        
        return self.factor_methods[factor_name](df)
    
    def get_factor_list(self) -> list:
        """获取所有可用因子名称列表"""
        return sorted(self.factor_methods.keys())
    
    # ==================== 因子计算方法 ====================
    
    def factor_001(self, df):
        """Volume weighted average price of 25 levels"""
        total_volume = np.sum([df[f'bv{i}'].values for i in range(25)], axis=0)
        total_value = np.sum([df[f'bp{i}'].values * df[f'bv{i}'].values for i in range(25)], axis=0)
        return np.where(total_volume != 0, total_value / total_volume, np.nan)

    def factor_002(self, df):
        """Ratio of total ask volume to total bid volume"""
        total_ask_volume = np.sum([df[f'av{i}'].values for i in range(25)], axis=0)
        total_bid_volume = np.sum([df[f'bv{i}'].values for i in range(25)], axis=0)
        return np.where(total_bid_volume != 0, total_ask_volume / total_bid_volume, np.nan)

    def factor_003(self, df):
        """Price difference normalized by the total price"""
        total_bid_price = np.sum([df[f'bp{i}'].values for i in range(20)], axis=0)
        total_ask_price = np.sum([df[f'ap{i}'].values for i in range(20)], axis=0)
        total_price = total_bid_price + total_ask_price
        price_diff = total_bid_price - total_ask_price
        return np.where(total_price != 0, price_diff / total_price, np.nan)

    def factor_004(self, df):
        """Mid price calculation"""
        return (df['bp0'].values + df['ap0'].values) / 2

    def factor_005(self, df):
        """Mid price difference over time"""
        mid_price = (df['bp0'].values + df['ap0'].values) / 2
        mid_price_diff = np.diff(mid_price, prepend=np.nan)
        return mid_price_diff

    def factor_006(self, df):
        """Bid and ask strength ratio for the first 5 levels"""
        ask_strength = np.sum([df[f'ap{i}'].values * df[f'av{i}'].values for i in range(5)], axis=0)
        bid_strength = np.sum([df[f'bp{i}'].values * df[f'bv{i}'].values for i in range(5)], axis=0)
        strength_ratio = (bid_strength - ask_strength) / (bid_strength + ask_strength)
        return strength_ratio

    def factor_007(self, df):
        """Absolute difference in total bid and ask volume"""
        total_ask_volume = np.sum([df[f'av{i}'].values for i in range(25)], axis=0)
        total_bid_volume = np.sum([df[f'bv{i}'].values for i in range(25)], axis=0)
        volume_diff = total_bid_volume - total_ask_volume
        return volume_diff

    def factor_008(self, df):
        """Standard deviation of price differences for the top 10 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(10)])
        ask_prices = np.array([df[f'ap{i}'].values for i in range(10)])
        price_diff = bid_prices - ask_prices
        price_diff_std = np.std(price_diff, axis=0)
        return price_diff_std

    def factor_009(self, df):
        """Mean price difference for the top 5 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(5)])
        ask_prices = np.array([df[f'ap{i}'].values for i in range(5)])
        price_diff = bid_prices - ask_prices
        price_diff_mean = np.mean(price_diff, axis=0)
        return price_diff_mean

    def factor_010(self, df):
        """Percentage change in bid to ask volume ratio for the first 15 levels"""
        volume_ratio = np.array([df[f'bv{i}'].values / (df[f'av{i}'].values + df[f'bv{i}'].values) for i in range(15)])
        volume_ratio_pct_change = pd.DataFrame(volume_ratio.T).pct_change(axis=1).mean(axis=1).values
        return volume_ratio_pct_change

    def factor_011(self, df):
        """Sum of bid volumes for the first 20 levels"""
        return np.sum([df[f'bv{i}'].values for i in range(20)], axis=0)

    def factor_012(self, df):
        """Sum of ask volumes for the first 20 levels"""
        return np.sum([df[f'av{i}'].values for i in range(20)], axis=0)

    def factor_013(self, df):
        """Standard deviation of price differences for the first 10 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(10)])
        ask_prices = np.array([df[f'ap{i}'].values for i in range(10)])
        price_diff = bid_prices - ask_prices
        return np.std(price_diff, axis=0)

    def factor_014(self, df):
        """Mean difference in price changes for 15 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(15)])
        ask_prices = np.array([df[f'ap{i}'].values for i in range(15)])
        price_diff = bid_prices - ask_prices
        price_diff_diff = np.diff(price_diff, axis=0)
        price_diff_diff_mean = np.mean(price_diff_diff, axis=0, where=~np.isnan(price_diff_diff), out=np.zeros_like(price_diff_diff[0]))
        return price_diff_diff_mean

    def factor_015(self, df):
        """Total bid strength for the first 5 levels"""
        return np.sum([df[f'bp{i}'].values * df[f'bv{i}'].values for i in range(5)], axis=0)

    def factor_016(self, df):
        """Total ask strength for the first 5 levels"""
        return np.sum([df[f'ap{i}'].values * df[f'av{i}'].values for i in range(5)], axis=0)

    def factor_017(self, df):
        """Max price difference for the first 10 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(10)])
        ask_prices = np.array([df[f'ap{i}'].values for i in range(10)])
        price_diff = bid_prices - ask_prices
        return np.max(price_diff, axis=0)

    def factor_018(self, df):
        """Standard deviation of combined volumes for the first 5 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(5)])
        ask_volumes = np.array([df[f'av{i}'].values for i in range(5)])
        total_volumes = bid_volumes + ask_volumes
        return np.std(total_volumes, axis=0)

    def factor_019(self, df):
        """Change rate of cumulative price differences for 25 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(25)])
        ask_prices = np.array([df[f'ap{i}'].values for i in range(25)])
        price_diff = bid_prices - ask_prices
        price_diff_cumsum = np.cumsum(price_diff, axis=0)
        price_diff_cumsum_diff = np.diff(price_diff_cumsum, axis=0, prepend=np.nan)
        return np.nanmean(price_diff_cumsum_diff, axis=0)

    def factor_020(self, df):
        """Skewness of combined volumes for the first 15 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(15)])
        ask_volumes = np.array([df[f'av{i}'].values for i in range(15)])
        total_volumes = bid_volumes + ask_volumes
        total_volumes_df = pd.DataFrame(total_volumes.T, index=df.index)
        return total_volumes_df.skew(axis=1).values

    def factor_021(self, df):
        """Skewness of total volumes for the first 25 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(25)])
        ask_volumes = np.array([df[f'av{i}'].values for i in range(25)])
        total_volumes = bid_volumes + ask_volumes
        total_sum = np.sum(total_volumes, axis=0)
        result = pd.Series(total_sum).skew()
        result_array = np.full(len(df), result)  # 返回与df长度相同的数组
        return result_array

    def factor_022(self, df):
        """Price change rate for the first 10 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(10)])
        ask_prices = np.array([df[f'ap{i}'].values for i in range(10)])
        price_diff = bid_prices - ask_prices
        price_diff_df = pd.DataFrame(price_diff.T, index=df.index)
        return price_diff_df.pct_change().mean(axis=1).values

    def factor_023(self, df):
        """Standard deviation difference of volumes for the first 20 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(20)])
        ask_volumes = np.array([df[f'av{i}'].values for i in range(20)])
        bid_volumes_std = np.std(bid_volumes, axis=0)
        ask_volumes_std = np.std(ask_volumes, axis=0)
        return bid_volumes_std - ask_volumes_std

    def factor_024(self, df):
        """Kurtosis of combined volumes for the first 5 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(5)])
        ask_volumes = np.array([df[f'av{i}'].values for i in range(5)])
        total_volumes = bid_volumes + ask_volumes
        total_volumes_df = pd.DataFrame(total_volumes.T, index=df.index)
        return total_volumes_df.kurt(axis=1).values

    def factor_025(self, df):
        """Standard deviation of volume ratios for the first 15 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(15)])
        ask_volumes = np.array([df[f'av{i}'].values for i in range(15)])
        volume_ratios = bid_volumes / (ask_volumes + bid_volumes)
        return np.std(volume_ratios, axis=0)

    def factor_026(self, df):
        """Standard deviation difference of prices for the first 10 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(10)])
        ask_prices = np.array([df[f'ap{i}'].values for i in range(10)])
        bid_prices_std = np.std(bid_prices, axis=0)
        ask_prices_std = np.std(ask_prices, axis=0)
        return bid_prices_std - ask_prices_std

    def factor_027(self, df):
        """Difference in bid and ask volume volatility for the first 25 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(25)])
        ask_volumes = np.array([df[f'av{i}'].values for i in range(25)])
        bid_volumes_df = pd.DataFrame(bid_volumes.T, index=df.index)
        ask_volumes_df = pd.DataFrame(ask_volumes.T, index=df.index)
        bid_volumes_diff_std = bid_volumes_df.diff().std(axis=1)
        ask_volumes_diff_std = ask_volumes_df.diff().std(axis=1)
        return (bid_volumes_diff_std - ask_volumes_diff_std).values

    def factor_028(self, df):
        """Log difference of bid and ask volumes for the first 25 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(25)])
        ask_volumes = np.array([df[f'av{i}'].values for i in range(25)])
        log_bid_volumes = np.log(bid_volumes + 1)
        log_ask_volumes = np.log(ask_volumes + 1)
        return np.mean(log_bid_volumes - log_ask_volumes, axis=0)

    def factor_029(self, df):
        """Mean price difference for the first 10 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(10)])
        ask_prices = np.array([df[f'ap{i}'].values for i in range(10)])
        bid_mean = np.mean(bid_prices, axis=0)
        ask_mean = np.mean(ask_prices, axis=0)
        return bid_mean - ask_mean

    def factor_030(self, df):
        """Volume ratio change for random 10 pairs"""
        np.random.seed(0)
        random_indices = np.random.choice(25, 10, replace=False)
        bid_volumes = np.array([df[f'bv{i}'].values for i in random_indices])
        ask_volumes = np.array([df[f'av{i}'].values for i in random_indices])
        volume_ratios = bid_volumes / (ask_volumes + bid_volumes)
        volume_ratios_df = pd.DataFrame(volume_ratios.T, index=df.index)
        return volume_ratios_df.pct_change().mean(axis=1).values

    def factor_031(self, df):
        """Max bid price for the first 5 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(5)])
        return np.max(bid_prices, axis=0)

    def factor_032(self, df):
        """Min ask price for the first 5 levels"""
        ask_prices = np.array([df[f'ap{i}'].values for i in range(5)])
        return np.min(ask_prices, axis=0)

    def factor_033(self, df):
        """Sum of bid volumes for the first 10 levels"""
        return np.sum([df[f'bv{i}'].values for i in range(10)], axis=0)

    def factor_034(self, df):
        """Sum of ask volumes for the first 10 levels"""
        return np.sum([df[f'av{i}'].values for i in range(10)], axis=0)

    def factor_035(self, df):
        """Mean bid price for the first 15 levels"""
        return np.mean([df[f'bp{i}'].values for i in range(15)], axis=0)

    def factor_036(self, df):
        """Mean ask price for the first 15 levels"""
        return np.mean([df[f'ap{i}'].values for i in range(15)], axis=0)

    def factor_037(self, df):
        """Sum of squared differences of bid prices for the first 10 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(10)])
        mean_bid_price = np.mean(bid_prices, axis=0)
        return np.sum((bid_prices - mean_bid_price) ** 2, axis=0)

    def factor_038(self, df):
        """Sum of squared differences of ask prices for the first 10 levels"""
        ask_prices = np.array([df[f'ap{i}'].values for i in range(10)])
        mean_ask_price = np.mean(ask_prices, axis=0)
        return np.sum((ask_prices - mean_ask_price) ** 2, axis=0)

    def factor_039(self, df):
        """Total volume difference for the first 25 levels"""
        total_bid_volume = np.sum([df[f'bv{i}'].values for i in range(25)], axis=0)
        total_ask_volume = np.sum([df[f'av{i}'].values for i in range(25)], axis=0)
        return total_bid_volume - total_ask_volume

    def factor_040(self, df):
        """Volatility of bid volumes for the first 10 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(10)])
        return np.std(bid_volumes, axis=0)

    def factor_041(self, df):
        """Volatility of ask volumes for the first 10 levels"""
        ask_volumes = np.array([df[f'av{i}'].values for i in range(10)])
        return np.std(ask_volumes, axis=0)

    def factor_042(self, df):
        """Ratio of bid volume to ask volume for the first 5 levels"""
        total_bid_volume = np.sum([df[f'bv{i}'].values for i in range(5)], axis=0)
        total_ask_volume = np.sum([df[f'av{i}'].values for i in range(5)], axis=0)
        return np.where(total_ask_volume != 0, total_bid_volume / total_ask_volume, np.nan)

    def factor_043(self, df):
        """Ratio of top 1 bid and ask volumes to the total volume in first 25 levels"""
        total_volume = np.sum([df[f'bv{i}'].values + df[f'av{i}'].values for i in range(25)], axis=0)
        top1_volume = df['bv0'].values + df['av0'].values
        return np.where(total_volume != 0, top1_volume / total_volume, np.nan)

    def factor_044(self, df):
        """Mean of the bid-ask spread for the first 15 levels"""
        spread = np.array([df[f'ap{i}'].values - df[f'bp{i}'].values for i in range(15)])
        return np.mean(spread, axis=0)

    def factor_045(self, df):
        """Rolling mean of bid volumes for the first 10 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(10)])
        mean_volumes = bid_volumes.mean(axis=0)
        rolling_result = pd.Series(mean_volumes).rolling(window=3).mean()
        return np.full(len(df), rolling_result.iloc[-1])

    def factor_046(self, df):
        """Rolling mean of ask volumes for the first 10 levels"""
        ask_volumes = np.array([df[f'av{i}'].values for i in range(10)])
        mean_volumes = ask_volumes.mean(axis=0)
        rolling_result = pd.Series(mean_volumes).rolling(window=3).mean()
        return np.full(len(df), rolling_result.iloc[-1])

    def factor_047(self, df):
        """Skewness of the ask prices for the first 15 levels"""
        ask_prices = np.array([df[f'ap{i}'].values for i in range(15)])
        result = pd.Series(ask_prices.mean(axis=0)).skew()
        return np.full(len(df), result)

    def factor_048(self, df):
        """Skewness of the bid prices for the first 15 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(15)])
        result = pd.Series(bid_prices.mean(axis=0)).skew()
        return np.full(len(df), result)

    def factor_049(self, df):
        """Rolling std of bid prices for the first 5 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(5)])
        mean_prices = bid_prices.mean(axis=0)
        rolling_result = pd.Series(mean_prices).rolling(window=3).std()
        return np.full(len(df), rolling_result.iloc[-1])

    def factor_050(self, df):
        """Rolling std of ask prices for the first 5 levels"""
        ask_prices = np.array([df[f'ap{i}'].values for i in range(5)])
        mean_prices = ask_prices.mean(axis=0)
        rolling_result = pd.Series(mean_prices).rolling(window=3).std()
        return np.full(len(df), rolling_result.iloc[-1])

    def factor_051(self, df):
        """Weighted average of bid prices with volumes for the first 5 levels"""
        weights = np.array([df[f'bv{i}'].values for i in range(5)])
        bid_prices = np.array([df[f'bp{i}'].values for i in range(5)])
        return np.average(bid_prices, weights=weights, axis=0)

    def factor_052(self, df):
        """Weighted average of ask prices with volumes for the first 5 levels"""
        weights = np.array([df[f'av{i}'].values for i in range(5)])
        ask_prices = np.array([df[f'ap{i}'].values for i in range(5)])
        return np.average(ask_prices, weights=weights, axis=0)

    def factor_053(self, df):
        """Ratio of top 1 bid and ask volumes to top 5 levels"""
        top1_bid_ask = df['bv0'].values + df['av0'].values
        top5_bid_ask = np.sum([df[f'bv{i}'].values + df[f'av{i}'].values for i in range(5)], axis=0)
        return np.where(top5_bid_ask != 0, top1_bid_ask / top5_bid_ask, np.nan)

    def factor_054(self, df):
        """Difference between mean of top 5 ask and bid prices"""
        mean_bid_5 = np.mean([df[f'bp{i}'].values for i in range(5)], axis=0)
        mean_ask_5 = np.mean([df[f'ap{i}'].values for i in range(5)], axis=0)
        return mean_ask_5 - mean_bid_5

    def factor_055(self, df):
        """Rolling sum of bid volumes for the first 10 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(10)])
        return pd.DataFrame(bid_volumes.T).rolling(window=3).sum().iloc[:, -1].values

    def factor_056(self, df):
        """Rolling sum of ask volumes for the first 10 levels"""
        ask_volumes = np.array([df[f'av{i}'].values for i in range(10)])
        return pd.DataFrame(ask_volumes.T).rolling(window=3).sum().iloc[:, -1].values

    def factor_057(self, df):
        """Total trade volume for the first 25 levels"""
        return np.sum([df[f'bv{i}'].values + df[f'av{i}'].values for i in range(25)], axis=0)

    def factor_058(self, df):
        """Cumulative sum of bid volumes for the first 10 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(10)])
        return np.cumsum(bid_volumes, axis=0)[-1]

    def factor_059(self, df):
        """Cumulative sum of ask volumes for the first 10 levels"""
        ask_volumes = np.array([df[f'av{i}'].values for i in range(10)])
        return np.cumsum(ask_volumes, axis=0)[-1]

    def factor_060(self, df):
        """Mean of bid volumes for the first 10 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(10)])
        return np.mean(bid_volumes, axis=0)

    def factor_061(self, df):
        """Mean of ask volumes for the first 10 levels"""
        ask_volumes = np.array([df[f'av{i}'].values for i in range(10)])
        return np.mean(ask_volumes, axis=0)

    def factor_062(self, df):
        """Weighted average price difference for the first 5 levels"""
        weights = np.array([df[f'bv{i}'].values + df[f'av{i}'].values for i in range(5)])
        price_diff = np.array([df[f'ap{i}'].values - df[f'bp{i}'].values for i in range(5)])
        return np.average(price_diff, weights=weights, axis=0)

    def factor_063(self, df):
        """Maximum volume of bids in top 10 levels"""
        return np.max([df[f'bv{i}'].values for i in range(10)], axis=0)

    def factor_064(self, df):
        """Maximum volume of asks in top 10 levels"""
        return np.max([df[f'av{i}'].values for i in range(10)], axis=0)

    def factor_065(self, df):
        """Minimum volume of bids in top 10 levels"""
        return np.min([df[f'bv{i}'].values for i in range(10)], axis=0)

    def factor_066(self, df):
        """Minimum volume of asks in top 10 levels"""
        return np.min([df[f'av{i}'].values for i in range(10)], axis=0)

    def factor_067(self, df):
        """Difference in cumulative sum of bid and ask volumes for top 10 levels"""
        bid_volumes = np.cumsum([df[f'bv{i}'].values for i in range(10)], axis=0)
        ask_volumes = np.cumsum([df[f'av{i}'].values for i in range(10)], axis=0)
        return bid_volumes[-1] - ask_volumes[-1]

    def factor_068(self, df):
        """Ratio of bid volume to ask volume for top 10 levels"""
        bid_volumes = np.sum([df[f'bv{i}'].values for i in range(10)], axis=0)
        ask_volumes = np.sum([df[f'av{i}'].values for i in range(10)], axis=0)
        return np.where(ask_volumes != 0, bid_volumes / ask_volumes, np.nan)

    def factor_069(self, df):
        """Sum of square root of bid volumes for the first 5 levels"""
        return np.sum([np.sqrt(df[f'bv{i}'].values) for i in range(5)], axis=0)

    def factor_070(self, df):
        """Sum of square root of ask volumes for the first 5 levels"""
        return np.sum([np.sqrt(df[f'av{i}'].values) for i in range(5)], axis=0)

    def factor_071(self, df):
        """Mean of square root of bid volumes for the first 5 levels"""
        return np.mean([np.sqrt(df[f'bv{i}'].values) for i in range(5)], axis=0)

    def factor_072(self, df):
        """Mean of square root of ask volumes for the first 5 levels"""
        return np.mean([np.sqrt(df[f'av{i}'].values) for i in range(5)], axis=0)

    def factor_073(self, df):
        """Standard deviation of bid volumes for the first 5 levels"""
        return np.std([df[f'bv{i}'].values for i in range(5)], axis=0)

    def factor_074(self, df):
        """Standard deviation of ask volumes for the first 5 levels"""
        return np.std([df[f'av{i}'].values for i in range(5)], axis=0)

    def factor_075(self, df):
        """Coefficient of variation of bid volumes for the first 5 levels"""
        mean_bid_volume = np.mean([df[f'bv{i}'].values for i in range(5)], axis=0)
        std_bid_volume = np.std([df[f'bv{i}'].values for i in range(5)], axis=0)
        return np.where(mean_bid_volume != 0, std_bid_volume / mean_bid_volume, np.nan)

    def factor_076(self, df):
        """Coefficient of variation of ask volumes for the first 5 levels"""
        mean_ask_volume = np.mean([df[f'av{i}'].values for i in range(5)], axis=0)
        std_ask_volume = np.std([df[f'av{i}'].values for i in range(5)], axis=0)
        return np.where(mean_ask_volume != 0, std_ask_volume / mean_ask_volume, np.nan)

    def factor_077(self, df):
        """Bid and ask volume correlation for top 10 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(10)])
        ask_volumes = np.array([df[f'av{i}'].values for i in range(10)])
        correlations = [np.corrcoef(bv, av)[0, 1] for bv, av in zip(bid_volumes.T, ask_volumes.T)]
        return np.array(correlations)

    def factor_078(self, df):
        """Ratio of sum of bid volumes to total volume of first 25 levels"""
        total_bid_volume = np.sum([df[f'bv{i}'].values for i in range(25)], axis=0)
        total_volume = np.sum([df[f'bv{i}'].values + df[f'av{i}'].values for i in range(25)], axis=0)
        return np.where(total_volume != 0, total_bid_volume / total_volume, np.nan)

    def factor_079(self, df):
        """Ratio of sum of ask volumes to total volume of first 25 levels"""
        total_ask_volume = np.sum([df[f'av{i}'].values for i in range(25)], axis=0)
        total_volume = np.sum([df[f'bv{i}'].values + df[f'av{i}'].values for i in range(25)], axis=0)
        return np.where(total_volume != 0, total_ask_volume / total_volume, np.nan)

    def factor_080(self, df):
        """Rolling std deviation of combined volumes for top 10 levels"""
        combined_volumes = np.array([df[f'bv{i}'].values + df[f'av{i}'].values for i in range(10)])
        return pd.DataFrame(combined_volumes.T).rolling(window=3).std().iloc[:, -1].values

    def factor_081(self, df):
        """Rolling mean of combined volumes for top 10 levels"""
        combined_volumes = np.array([df[f'bv{i}'].values + df[f'av{i}'].values for i in range(10)])
        return pd.DataFrame(combined_volumes.T).rolling(window=3).mean().iloc[:, -1].values

    def factor_082(self, df):
        """Rolling sum of combined volumes for top 10 levels"""
        combined_volumes = np.array([df[f'bv{i}'].values + df[f'av{i}'].values for i in range(10)])
        return pd.DataFrame(combined_volumes.T).rolling(window=3).sum().iloc[:, -1].values

    def factor_083(self, df):
        """Difference between max and min of bid volumes for top 10 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(10)])
        return np.max(bid_volumes, axis=0) - np.min(bid_volumes, axis=0)

    def factor_084(self, df):
        """Difference between max and min of ask volumes for top 10 levels"""
        ask_volumes = np.array([df[f'av{i}'].values for i in range(10)])
        return np.max(ask_volumes, axis=0) - np.min(ask_volumes, axis=0)

    def factor_085(self, df):
        """Covariance of bid and ask volumes for top 10 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(10)])
        ask_volumes = np.array([df[f'av{i}'].values for i in range(10)])
        covariances = [np.cov(bv, av)[0, 1] for bv, av in zip(bid_volumes.T, ask_volumes.T)]
        return np.array(covariances)

    def factor_086(self, df):
        """Difference between sum of bid and ask volumes for top 15 levels"""
        total_bid_volume = np.sum([df[f'bv{i}'].values for i in range(15)], axis=0)
        total_ask_volume = np.sum([df[f'av{i}'].values for i in range(15)], axis=0)
        return total_bid_volume - total_ask_volume

    def factor_087(self, df):
        """Ratio of sum of bid volumes to ask volumes for top 20 levels"""
        total_bid_volume = np.sum([df[f'bv{i}'].values for i in range(20)], axis=0)
        total_ask_volume = np.sum([df[f'av{i}'].values for i in range(20)], axis=0)
        return np.where(total_ask_volume != 0, total_bid_volume / total_ask_volume, np.nan)

    def factor_088(self, df):
        """Variance of bid volumes for the first 10 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(10)])
        return np.var(bid_volumes, axis=0)

    def factor_089(self, df):
        """Variance of ask volumes for the first 10 levels"""
        ask_volumes = np.array([df[f'av{i}'].values for i in range(10)])
        return np.var(ask_volumes, axis=0)

    def factor_090(self, df):
        """Average of bid and ask price difference for top 10 levels"""
        price_diff = np.array([df[f'ap{i}'].values - df[f'bp{i}'].values for i in range(10)])
        return np.mean(price_diff, axis=0)

    def factor_091(self, df):
        """Volatility ratio of bid to ask volumes for top 10 levels"""
        bid_volatility = np.std([df[f'bv{i}'].values for i in range(10)], axis=0)
        ask_volatility = np.std([df[f'av{i}'].values for i in range(10)], axis=0)
        return np.where(ask_volatility != 0, bid_volatility / ask_volatility, np.nan)

    def factor_092(self, df):
        """Volatility ratio of bid to ask prices for top 10 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(10)])
        ask_prices = np.array([df[f'ap{i}'].values for i in range(10)])
        bid_volatility = np.std(bid_prices, axis=0)
        ask_volatility = np.std(ask_prices, axis=0)
        return np.where(ask_volatility != 0, bid_volatility / ask_volatility, np.nan)

    def factor_093(self, df):
        """Mean of bid prices for the first 5 levels"""
        return np.mean([df[f'bp{i}'].values for i in range(5)], axis=0)

    def factor_094(self, df):
        """Mean of ask prices for the first 5 levels"""
        return np.mean([df[f'ap{i}'].values for i in range(5)], axis=0)

    def factor_095(self, df):
        """Median of bid volumes for the first 25 levels"""
        return np.median([df[f'bv{i}'].values for i in range(25)], axis=0)

    def factor_096(self, df):
        """Median of ask volumes for the first 25 levels"""
        return np.median([df[f'av{i}'].values for i in range(25)], axis=0)

    def factor_097(self, df):
        """Ratio of bid volumes to ask volumes for each level, averaged over the first 10 levels"""
        ratios = np.array([df[f'bv{i}'].values / df[f'av{i}'].values if df[f'av{i}'].values.all() != 0 else np.nan for i in range(10)])
        return np.nanmean(ratios, axis=0)

    def factor_098(self, df):
        """Range (max-min) of bid prices for the first 15 levels"""
        return np.max([df[f'bp{i}'].values for i in range(15)], axis=0) - np.min([df[f'bp{i}'].values for i in range(15)], axis=0)

    def factor_099(self, df):
        """Range (max-min) of ask prices for the first 15 levels"""
        return np.max([df[f'ap{i}'].values for i in range(15)], axis=0) - np.min([df[f'ap{i}'].values for i in range(15)], axis=0)

    def factor_100(self, df):
        """Sum of the square of the differences of bid prices for the first 10 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(10)])
        return np.sum((bid_prices - np.mean(bid_prices, axis=0)) ** 2, axis=0)

    def factor_101(self, df):
        """Sum of the square of the differences of ask prices for the first 10 levels"""
        ask_prices = np.array([df[f'ap{i}'].values for i in range(10)])
        return np.sum((ask_prices - np.mean(ask_prices, axis=0)) ** 2, axis=0)

    def factor_102(self, df):
        """Harmonic mean of bid volumes for the first 10 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(10)])
        return np.where(np.all(bid_volumes, axis=0), 10 / np.sum(1.0 / bid_volumes, axis=0), np.nan)

    def factor_103(self, df):
        """Harmonic mean of ask volumes for the first 10 levels"""
        ask_volumes = np.array([df[f'av{i}'].values for i in range(10)])
        return np.where(np.all(ask_volumes, axis=0), 10 / np.sum(1.0 / ask_volumes, axis=0), np.nan)

    def factor_104(self, df):
        """Weighted mean of bid prices by ask volumes for the first 5 levels"""
        weights = np.array([df[f'av{i}'].values for i in range(5)])
        bid_prices = np.array([df[f'bp{i}'].values for i in range(5)])
        return np.average(bid_prices, weights=weights, axis=0)

    def factor_105(self, df):
        """Weighted mean of ask prices by bid volumes for the first 5 levels"""
        weights = np.array([df[f'bv{i}'].values for i in range(5)])
        ask_prices = np.array([df[f'ap{i}'].values for i in range(5)])
        return np.average(ask_prices, weights=weights, axis=0)

    def factor_106(self, df):
        """Price range between the maximum bid and minimum ask in the first 20 levels"""
        max_bid_price = np.max(np.array([df[f'bp{i}'].values for i in range(20)]), axis=0)
        min_ask_price = np.min(np.array([df[f'ap{i}'].values for i in range(20)]), axis=0)
        return max_bid_price - min_ask_price

    def factor_107(self, df):
        """Correlation between bid prices and volumes for the first 10 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(10)])
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(10)])
        correlations = np.array([np.corrcoef(bid_prices[:, i], bid_volumes[:, i])[0, 1] for i in range(bid_prices.shape[1])])
        return correlations

    def factor_108(self, df):
        """Correlation between ask prices and volumes for the first 10 levels"""
        ask_prices = np.array([df[f'ap{i}'].values for i in range(10)])
        ask_volumes = np.array([df[f'av{i}'].values for i in range(10)])
        correlations = np.array([np.corrcoef(ask_prices[:, i], ask_volumes[:, i])[0, 1] for i in range(ask_prices.shape[1])])
        return correlations

    def factor_109(self, df):
        """Average difference between bid prices and the mid-price for the first 5 levels"""
        mid_price = (df['bp0'].values + df['ap0'].values) / 2
        bid_prices = np.array([df[f'bp{i}'].values for i in range(5)])
        return np.mean(bid_prices - mid_price, axis=0)

    def factor_110(self, df):
        """Average difference between ask prices and the mid-price for the first 5 levels"""
        mid_price = (df['bp0'].values + df['ap0'].values) / 2
        ask_prices = np.array([df[f'ap{i}'].values for i in range(5)])
        return np.mean(ask_prices - mid_price, axis=0)

    def factor_111(self, df):
        """Rolling variance of bid prices for the first 10 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(10)]).T
        rolling_variance = pd.DataFrame(bid_prices).rolling(window=3).var().values.T
        return rolling_variance[-1]

    def factor_112(self, df):
        """Rolling variance of ask prices for the first 10 levels"""
        ask_prices = np.array([df[f'ap{i}'].values for i in range(10)]).T
        rolling_variance = pd.DataFrame(ask_prices).rolling(window=3).var().values.T
        return rolling_variance[-1]

    def factor_113(self, df):
        """Harmonic mean of bid prices for the first 10 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(10)])
        return np.where(np.all(bid_prices, axis=0), 10 / np.sum(1.0 / bid_prices, axis=0), np.nan)

    def factor_114(self, df):
        """Harmonic mean of ask prices for the first 10 levels"""
        ask_prices = np.array([df[f'ap{i}'].values for i in range(10)])
        return np.where(np.all(ask_prices, axis=0), 10 / np.sum(1.0 / ask_prices, axis=0), np.nan)

    def factor_115(self, df):
        """Difference in harmonic mean of bid and ask prices for the first 10 levels"""
        return self.factor_113(df) - self.factor_114(df)

    def factor_116(self, df):
        """Weighted mean of bid prices with inverse volumes for the first 10 levels"""
        weights = 1 / np.array([df[f'bv{i}'].values for i in range(10)])
        bid_prices = np.array([df[f'bp{i}'].values for i in range(10)])
        return np.average(bid_prices, weights=weights, axis=0)

    def factor_117(self, df):
        """Weighted mean of ask prices with inverse volumes for the first 10 levels"""
        weights = 1 / np.array([df[f'av{i}'].values for i in range(10)])
        ask_prices = np.array([df[f'ap{i}'].values for i in range(10)])
        return np.average(ask_prices, weights=weights, axis=0)

    def factor_118(self, df):
        """Maximum bid price difference in the first 15 levels"""
        return np.max(np.diff(np.array([df[f'bp{i}'].values for i in range(15)]), axis=0), axis=0)

    def factor_119(self, df):
        """Maximum ask price difference in the first 15 levels"""
        return np.max(np.diff(np.array([df[f'ap{i}'].values for i in range(15)]), axis=0), axis=0)

    def factor_120(self, df):
        """Median bid volume for the first 10 levels"""
        return np.median(np.array([df[f'bv{i}'].values for i in range(10)]), axis=0)

    def factor_121(self, df):
        """Median ask volume for the first 10 levels"""
        return np.median(np.array([df[f'av{i}'].values for i in range(10)]), axis=0)

    def factor_122(self, df):
        """Top 3 ask volumes to total ask volume ratio in the first 15 levels"""
        top3_ask_volumes = np.sum(np.array([df[f'av{i}'].values for i in range(3)]), axis=0)
        total_ask_volumes = np.sum(np.array([df[f'av{i}'].values for i in range(15)]), axis=0)
        return np.where(total_ask_volumes != 0, top3_ask_volumes / total_ask_volumes, np.nan)

    def factor_123(self, df):
        """Top 3 bid volumes to total bid volume ratio in the first 15 levels"""
        top3_bid_volumes = np.sum(np.array([df[f'bv{i}'].values for i in range(3)]), axis=0)
        total_bid_volumes = np.sum(np.array([df[f'bv{i}'].values for i in range(15)]), axis=0)
        return np.where(total_bid_volumes != 0, top3_bid_volumes / total_bid_volumes, np.nan)

    def factor_124(self, df):
        """Ratio of variance of bid and ask volumes for the first 10 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(10)])
        ask_volumes = np.array([df[f'av{i}'].values for i in range(10)])
        bid_variance = np.var(bid_volumes, axis=0)
        ask_variance = np.var(ask_volumes, axis=0)
        return np.where(ask_variance != 0, bid_variance / ask_variance, np.nan)

    def factor_125(self, df):
        """Median of prices between bid and ask for the first 20 levels"""
        prices = np.array([df[f'bp{i}'].values for i in range(20)] + [df[f'ap{i}'].values for i in range(20)])
        return np.median(prices, axis=0)

    def factor_126(self, df):
        """Coefficient of variation of combined volumes for top 10 levels"""
        combined_volumes = np.array([df[f'bv{i}'].values + df[f'av{i}'].values for i in range(10)])
        mean_combined_volume = np.mean(combined_volumes, axis=0)
        std_combined_volume = np.std(combined_volumes, axis=0)
        return np.where(mean_combined_volume != 0, std_combined_volume / mean_combined_volume, np.nan)

    def factor_127(self, df):
        """Mean of ask prices weighted by bid volumes for the first 10 levels"""
        weights = np.array([df[f'bv{i}'].values for i in range(10)])
        ask_prices = np.array([df[f'ap{i}'].values for i in range(10)])
        return np.average(ask_prices, weights=weights, axis=0)

    def factor_128(self, df):
        """Mean of bid prices weighted by ask volumes for the first 10 levels"""
        weights = np.array([df[f'av{i}'].values for i in range(10)])
        bid_prices = np.array([df[f'bp{i}'].values for i in range(10)])
        return np.average(bid_prices, weights=weights, axis=0)

    def factor_129(self, df):
        """Weighted harmonic mean of bid volumes for the first 10 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(10)])
        weights = np.arange(1, 11)[:, np.newaxis]
        valid_mask = np.all(bid_volumes > 0, axis=0)
        harmonic_means = np.full(len(df), np.nan)
        harmonic_means[valid_mask] = len(weights) / np.sum(weights / bid_volumes[:, valid_mask], axis=0)
        return harmonic_means

    def factor_130(self, df):
        """Weighted harmonic mean of ask volumes for the first 10 levels"""
        ask_volumes = np.array([df[f'av{i}'].values for i in range(10)])
        weights = np.arange(1, 11)[:, np.newaxis]
        valid_mask = np.all(ask_volumes > 0, axis=0)
        harmonic_means = np.full(len(df), np.nan)
        harmonic_means[valid_mask] = len(weights) / np.sum(weights / ask_volumes[:, valid_mask], axis=0)
        return harmonic_means

    def factor_131(self, df):
        """Volume weighted average price of 25 levels"""
        total_volume = np.sum([df[f'bv{i}'].values for i in range(25)], axis=0)
        total_value = np.sum([df[f'bp{i}'].values * df[f'bv{i}'].values for i in range(25)], axis=0)
        return np.where(total_volume != 0, total_value / total_volume, np.nan)

    def factor_132(self, df):
        """Ratio of total ask volume to total bid volume"""
        total_ask_volume = np.sum([df[f'av{i}'].values for i in range(25)], axis=0)
        total_bid_volume = np.sum([df[f'bv{i}'].values for i in range(25)], axis=0)
        return np.where(total_bid_volume != 0, total_ask_volume / total_bid_volume, np.nan)

    def factor_133(self, df):
        """Price difference normalized by the total price"""
        total_bid_price = np.sum([df[f'bp{i}'].values for i in range(20)], axis=0)
        total_ask_price = np.sum([df[f'ap{i}'].values for i in range(20)], axis=0)
        total_price = total_bid_price + total_ask_price
        price_diff = total_bid_price - total_ask_price
        return np.where(total_price != 0, price_diff / total_price, np.nan)

    def factor_134(self, df):
        """Mid price calculation"""
        return (df['bp0'].values + df['ap0'].values) / 2

    def factor_135(self, df):
        """Mid price difference over time"""
        center_price = (df['bp0'].values + df['ap0'].values) / 2
        return np.diff(center_price, prepend=np.nan)

    def factor_136(self, df):
        """Bid and ask strength ratio for the first 5 levels"""
        ask_strength = np.sum([df[f'ap{i}'].values * df[f'av{i}'].values for i in range(5)], axis=0)
        bid_strength = np.sum([df[f'bp{i}'].values * df[f'bv{i}'].values for i in range(5)], axis=0)
        return (bid_strength - ask_strength) / (bid_strength + ask_strength)

    def factor_137(self, df):
        """Absolute difference in total bid and ask volume"""
        total_ask_volume = np.sum([df[f'av{i}'].values for i in range(25)], axis=0)
        total_bid_volume = np.sum([df[f'bv{i}'].values for i in range(25)], axis=0)
        return total_bid_volume - total_ask_volume

    def factor_138(self, df):
        """Standard deviation of price differences for the top 10 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(10)])
        ask_prices = np.array([df[f'ap{i}'].values for i in range(10)])
        price_diff = bid_prices - ask_prices
        return np.std(price_diff, axis=0)

    def factor_139(self, df):
        """Mean price difference for the top 5 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(5)])
        ask_prices = np.array([df[f'ap{i}'].values for i in range(5)])
        price_diff = bid_prices - ask_prices
        return np.mean(price_diff, axis=0)

    def factor_140(self, df):
        """Percentage change in bid to ask volume ratio for the first 15 levels"""
        volume_ratio = np.array([df[f'bv{i}'].values / (df[f'av{i}'].values + df[f'bv{i}'].values) for i in range(15)])
        volume_ratio_df = pd.DataFrame(volume_ratio.T)
        return volume_ratio_df.pct_change(axis=1).mean(axis=1).values

    def factor_141(self, df):
        """Sum of bid volumes for the first 20 levels"""
        return np.sum([df[f'bv{i}'].values for i in range(20)], axis=0)

    def factor_142(self, df):
        """Sum of ask volumes for the first 20 levels"""
        return np.sum([df[f'av{i}'].values for i in range(20)], axis=0)

    def factor_143(self, df):
        """Standard deviation of price differences for the first 10 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(10)])
        ask_prices = np.array([df[f'ap{i}'].values for i in range(10)])
        price_diff = bid_prices - ask_prices
        return np.std(price_diff, axis=0)

    def factor_144(self, df):
        """Mean difference in price changes for 15 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(15)])
        ask_prices = np.array([df[f'ap{i}'].values for i in range(15)])
        price_diff = bid_prices - ask_prices
        price_diff_diff = np.diff(price_diff, axis=0, prepend=np.nan)
        return np.nanmean(price_diff_diff, axis=0)

    def factor_145(self, df):
        """Total bid strength for the first 5 levels"""
        return np.sum([df[f'bp{i}'].values * df[f'bv{i}'].values for i in range(5)], axis=0)

    def factor_146(self, df):
        """Total ask strength for the first 5 levels"""
        return np.sum([df[f'ap{i}'].values * df[f'av{i}'].values for i in range(5)], axis=0)

    def factor_147(self, df):
        """Max price difference for the first 10 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(10)])
        ask_prices = np.array([df[f'ap{i}'].values for i in range(10)])
        price_diff = bid_prices - ask_prices
        return np.max(price_diff, axis=0)

    def factor_148(self, df):
        """Standard deviation of combined volumes for the first 5 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(5)])
        ask_volumes = np.array([df[f'av{i}'].values for i in range(5)])
        total_volumes = bid_volumes + ask_volumes
        return np.std(total_volumes, axis=0)

    def factor_149(self, df):
        """Change rate of cumulative price differences for 25 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(25)])
        ask_prices = np.array([df[f'ap{i}'].values for i in range(25)])
        price_diff = bid_prices - ask_prices
        price_diff_cumsum = np.cumsum(price_diff, axis=0)
        price_diff_cumsum_diff = np.diff(price_diff_cumsum, axis=0, prepend=np.nan)
        return np.nanmean(price_diff_cumsum_diff, axis=0)

    def factor_150(self, df):
        """Skewness of combined volumes for the first 15 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(15)])
        ask_volumes = np.array([df[f'av{i}'].values for i in range(15)])
        total_volumes = bid_volumes + ask_volumes
        total_volumes_df = pd.DataFrame(total_volumes.T, index=df.index)
        return total_volumes_df.skew(axis=1).values

    def factor_151(self, df):
        """Skewness of total volumes for the first 25 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(25)])
        ask_volumes = np.array([df[f'av{i}'].values for i in range(25)])
        total_volumes = bid_volumes + ask_volumes
        total_sum = np.sum(total_volumes, axis=0)
        result = pd.Series(total_sum).skew()
        result_array = np.full(len(df), result)
        return result_array

    def factor_152(self, df):
        """Price change rate for the first 10 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(10)])
        ask_prices = np.array([df[f'ap{i}'].values for i in range(10)])
        price_diff = bid_prices - ask_prices
        price_diff_df = pd.DataFrame(price_diff.T, index=df.index)
        return price_diff_df.pct_change().mean(axis=1).values

    def factor_153(self, df):
        """Standard deviation difference of volumes for the first 20 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(20)])
        ask_volumes = np.array([df[f'av{i}'].values for i in range(20)])
        bid_volumes_std = np.std(bid_volumes, axis=0)
        ask_volumes_std = np.std(ask_volumes, axis=0)
        return bid_volumes_std - ask_volumes_std

    def factor_154(self, df):
        """Kurtosis of combined volumes for the first 5 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(5)])
        ask_volumes = np.array([df[f'av{i}'].values for i in range(5)])
        total_volumes = bid_volumes + ask_volumes
        total_volumes_df = pd.DataFrame(total_volumes.T, index=df.index)
        return total_volumes_df.kurt(axis=1).values

    def factor_155(self, df):
        """Standard deviation of volume ratios for the first 15 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(15)])
        ask_volumes = np.array([df[f'av{i}'].values for i in range(15)])
        volume_ratios = bid_volumes / (ask_volumes + bid_volumes)
        return np.std(volume_ratios, axis=0)

    def factor_156(self, df):
        """Standard deviation difference of prices for the first 10 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(10)])
        ask_prices = np.array([df[f'ap{i}'].values for i in range(10)])
        bid_prices_std = np.std(bid_prices, axis=0)
        ask_prices_std = np.std(ask_prices, axis=0)
        return bid_prices_std - ask_prices_std

    def factor_157(self, df):
        """Difference in bid and ask volume volatility for the first 25 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(25)])
        ask_volumes = np.array([df[f'av{i}'].values for i in range(25)])
        bid_volumes_df = pd.DataFrame(bid_volumes.T, index=df.index)
        ask_volumes_df = pd.DataFrame(ask_volumes.T, index=df.index)
        bid_volumes_diff_std = bid_volumes_df.diff().std(axis=1).values
        ask_volumes_diff_std = ask_volumes_df.diff().std(axis=1).values
        return bid_volumes_diff_std - ask_volumes_diff_std

    def factor_158(self, df):
        """Log difference of bid and ask volumes for the first 25 levels"""
        bid_volumes = np.array([df[f'bv{i}'].values for i in range(25)])
        ask_volumes = np.array([df[f'av{i}'].values for i in range(25)])
        log_bid_volumes = np.log(bid_volumes + 1)
        log_ask_volumes = np.log(ask_volumes + 1)
        return np.mean(log_bid_volumes - log_ask_volumes, axis=0)

    def factor_159(self, df):
        """Mean price difference for the first 10 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(10)])
        ask_prices = np.array([df[f'ap{i}'].values for i in range(10)])
        bid_mean = np.mean(bid_prices, axis=0)
        ask_mean = np.mean(ask_prices, axis=0)
        return bid_mean - ask_mean

    def factor_160(self, df):
        """Volume ratio change for random 10 pairs"""
        np.random.seed(0)
        random_indices = np.random.choice(25, 10, replace=False)
        bid_volumes = np.array([df[f'bv{i}'].values for i in random_indices])
        ask_volumes = np.array([df[f'av{i}'].values for i in random_indices])
        volume_ratios = bid_volumes / (ask_volumes + bid_volumes)
        volume_ratios_df = pd.DataFrame(volume_ratios.T, index=df.index)
        return volume_ratios_df.pct_change().mean(axis=1).values


# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    import numpy as np
    import pandas as pd
    
    np.random.seed(42)
    n_rows = 1000
    
    # 创建示例数据
    data = {}
    for i in range(25):
        data[f'bp{i}'] = np.random.uniform(99, 101, n_rows)  # 买入价格
        data[f'ap{i}'] = np.random.uniform(101, 103, n_rows)  # 卖出价格
        data[f'bv{i}'] = np.random.uniform(1, 100, n_rows)   # 买入成交量
        data[f'av{i}'] = np.random.uniform(1, 100, n_rows)   # 卖出成交量
    
    df = pd.DataFrame(data)
    
    # 使用因子计算器
    calculator = AlphaFactorCalculator()
    
    # 计算所有因子
    result = calculator.calculate_all_factors(df)
    print(f"计算完成，共计算了 {len(result.columns)} 个因子")
    print(f"结果DataFrame形状: {result.shape}")
    print(f"因子列表: {result.columns.tolist()}")
    
    # 计算单个因子示例
    factor_001_result = calculator.calculate_single_factor(df, 'factor_001')
    print(f"Factor 001 前5个值: {factor_001_result[:5]}")
    
    # 获取所有可用因子列表
    available_factors = calculator.get_factor_list()
    print(f"可用因子数量: {len(available_factors)}")