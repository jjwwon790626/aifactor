import numpy as np
import pandas as pd


class SelectedAlphaFactorCalculator:
    """精简版Alpha因子计算器 - 只计算指定的23个因子"""
    
    def __init__(self):
        self.target_factors = [
            'factor_048', 'factor_047', 'factor_117', 'factor_031', 'factor_105', 
            'factor_134', 'factor_004', 'factor_125', 'factor_104', 'factor_094', 
            'factor_114', 'factor_127', 'factor_032', 'factor_051', 'factor_128', 
            'factor_052', 'factor_036', 'factor_093', 'factor_116', 'factor_113', 
            'factor_001', 'factor_131', 'factor_035'
        ]
    
    def calculate_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有指定因子"""
        factors_data = {}
        
        for factor_name in self.target_factors:
            try:
                factors_data[factor_name] = getattr(self, factor_name)(df)
            except Exception as e:
                print(f"计算 {factor_name} 时出错: {str(e)}")
                factors_data[factor_name] = np.full(len(df), np.nan)
        
        return pd.DataFrame(factors_data, index=df.index)
    
    # ==================== 因子计算方法 ====================
    
    def factor_001(self, df):
        """Volume weighted average price of 25 levels"""
        total_volume = np.sum([df[f'bv{i}'].values for i in range(25)], axis=0)
        total_value = np.sum([df[f'bp{i}'].values * df[f'bv{i}'].values for i in range(25)], axis=0)
        return np.where(total_volume != 0, total_value / total_volume, np.nan)

    def factor_004(self, df):
        """Mid price calculation"""
        return (df['bp0'].values + df['ap0'].values) / 2

    def factor_031(self, df):
        """Max bid price for the first 5 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(5)])
        return np.max(bid_prices, axis=0)

    def factor_032(self, df):
        """Min ask price for the first 5 levels"""
        ask_prices = np.array([df[f'ap{i}'].values for i in range(5)])
        return np.min(ask_prices, axis=0)

    def factor_035(self, df):
        """Mean bid price for the first 15 levels"""
        return np.mean([df[f'bp{i}'].values for i in range(15)], axis=0)

    def factor_036(self, df):
        """Mean ask price for the first 15 levels"""
        return np.mean([df[f'ap{i}'].values for i in range(15)], axis=0)

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

    def factor_093(self, df):
        """Mean of bid prices for the first 5 levels"""
        return np.mean([df[f'bp{i}'].values for i in range(5)], axis=0)

    def factor_094(self, df):
        """Mean of ask prices for the first 5 levels"""
        return np.mean([df[f'ap{i}'].values for i in range(5)], axis=0)

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

    def factor_113(self, df):
        """Harmonic mean of bid prices for the first 10 levels"""
        bid_prices = np.array([df[f'bp{i}'].values for i in range(10)])
        return np.where(np.all(bid_prices > 0, axis=0), 10 / np.sum(1.0 / bid_prices, axis=0), np.nan)

    def factor_114(self, df):
        """Harmonic mean of ask prices for the first 10 levels"""
        ask_prices = np.array([df[f'ap{i}'].values for i in range(10)])
        return np.where(np.all(ask_prices > 0, axis=0), 10 / np.sum(1.0 / ask_prices, axis=0), np.nan)

    def factor_116(self, df):
        """Weighted mean of bid prices with inverse volumes for the first 10 levels"""
        volumes = np.array([df[f'bv{i}'].values for i in range(10)])
        weights = np.where(volumes > 0, 1.0 / volumes, 0)
        bid_prices = np.array([df[f'bp{i}'].values for i in range(10)])
        return np.average(bid_prices, weights=weights, axis=0)

    def factor_117(self, df):
        """Weighted mean of ask prices with inverse volumes for the first 10 levels"""
        volumes = np.array([df[f'av{i}'].values for i in range(10)])
        weights = np.where(volumes > 0, 1.0 / volumes, 0)
        ask_prices = np.array([df[f'ap{i}'].values for i in range(10)])
        return np.average(ask_prices, weights=weights, axis=0)

    def factor_125(self, df):
        """Median of prices between bid and ask for the first 20 levels"""
        prices = np.array([df[f'bp{i}'].values for i in range(20)] + [df[f'ap{i}'].values for i in range(20)])
        return np.median(prices, axis=0)

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

    def factor_131(self, df):
        """Volume weighted average price of 25 levels"""
        total_volume = np.sum([df[f'bv{i}'].values for i in range(25)], axis=0)
        total_value = np.sum([df[f'bp{i}'].values * df[f'bv{i}'].values for i in range(25)], axis=0)
        return np.where(total_volume != 0, total_value / total_volume, np.nan)

    def factor_134(self, df):
        """Mid price calculation"""
        return (df['bp0'].values + df['ap0'].values) / 2