import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms, gp
from scipy import stats
import operator
import random

class FactorMiner:
    def __init__(self, data_dict):
        """
        初始化因子挖掘器
        
        参数:
        - data_dict: dict, 包含不同数据类型的DataFrame字典
            {
                'open': df_open,
                'high': df_high,
                'low': df_low,
                'close': df_close,
                'volume': df_volume
            }
        """
        self.data = data_dict
        self.returns = self.data['close'].pct_change()
        
        # 初始化遗传编程工具箱
        self._init_toolbox()
        
    def _init_toolbox(self):
        """初始化DEAP工具箱"""
        # 定义适应度和个体类型
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        
        # 定义基本运算符
        pset = gp.PrimitiveSetTyped("MAIN", 
                                   [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], 
                                   pd.DataFrame)  # 5个输入：OHLCV，输出为DataFrame
        
        # 定义基本运算符
        pset.addPrimitive(operator.add, [pd.DataFrame, pd.DataFrame], pd.DataFrame)
        pset.addPrimitive(operator.sub, [pd.DataFrame, pd.DataFrame], pd.DataFrame)
        pset.addPrimitive(operator.mul, [pd.DataFrame, pd.DataFrame], pd.DataFrame)
        pset.addPrimitive(operator.truediv, [pd.DataFrame, pd.DataFrame], pd.DataFrame)
        
        # 定义技术指标算子
        pset.addPrimitive(self.ts_mean, [pd.DataFrame, int], pd.DataFrame)  # 第一个参数是DataFrame，第二个是int
        pset.addPrimitive(self.ts_std, [pd.DataFrame, int], pd.DataFrame)
        pset.addPrimitive(self.ts_skew, [pd.DataFrame, int], pd.DataFrame)
        pset.addPrimitive(self.ts_kurt, [pd.DataFrame, int], pd.DataFrame)
        pset.addPrimitive(self.ts_max, [pd.DataFrame, int], pd.DataFrame)
        pset.addPrimitive(self.ts_min, [pd.DataFrame, int], pd.DataFrame)
        
        # 添加int类型的原始操作
        pset.addPrimitive(operator.add, [int, int], int)
        
        # 添加常量
        pset.addEphemeralConstant("lookback", lambda: random.randint(5, 20), int)  # 指定返回类型为int
        
        # 添加int类型的终端节点
        # for i in range(5, 21):  # 添加5到20的整数作为终端节点
        #     pset.addTerminal(i, int)
        
        # 重命名参数
        pset.renameArguments(ARG0='open')
        pset.renameArguments(ARG1='high')
        pset.renameArguments(ARG2='low')
        pset.renameArguments(ARG3='close')
        pset.renameArguments(ARG4='volume')
        
        # 注册遗传操作
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)  # 减小最大深度
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=pset)
        
        # 注册遗传算法操作
        self.toolbox.register("evaluate", self.evaluate_factor)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=pset)
        
        # 设置突变概率和深度限制
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))  # 限制最大深度为5
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))  # 限制最大深度为5
    
    # 技术指标算子
    def ts_mean(self, x: pd.DataFrame, lookback: int) -> pd.DataFrame:
        return pd.DataFrame(x).rolling(lookback).mean()
    
    def ts_std(self, x: pd.DataFrame, lookback: int) -> pd.DataFrame:
        return pd.DataFrame(x).rolling(lookback).std()
    
    def ts_skew(self, x: pd.DataFrame, lookback: int) -> pd.DataFrame:
        return pd.DataFrame(x).rolling(lookback).skew()
    
    def ts_kurt(self, x: pd.DataFrame, lookback: int) -> pd.DataFrame:
        return pd.DataFrame(x).rolling(lookback).kurt()
    
    def ts_max(self, x: pd.DataFrame, lookback: int) -> pd.DataFrame:
        return pd.DataFrame(x).rolling(lookback).max()
    
    def ts_min(self, x: pd.DataFrame, lookback: int) -> pd.DataFrame:
        return pd.DataFrame(x).rolling(lookback).min()
    
    def evaluate_factor(self, individual):
        """评估因子的IC值"""
        try:
            # 编译表达式
            func = self.toolbox.compile(expr=individual)
            
            # 计算因子值
            factor_value = func(
                self.data['open'],
                self.data['high'],
                self.data['low'],
                self.data['close'],
                self.data['volume']
            )
            forward_returns = self.returns.shift(-1)
            index = factor_value.dropna().index.intersection(forward_returns.dropna().index)
            # 计算下期收益
            forward_returns = self.returns.shift(-1).loc[index]
            factor_value = factor_value.loc[index]
            
            # 计算IC值
            ic_series = pd.DataFrame(factor_value).corrwith(
                forward_returns, method='spearman', axis=1
            )
            
            # 使用IC均值作为适应度
            ic_mean = ic_series.mean()
            
            return ic_mean,
            
        except Exception as e:
            return -999999,
    
    def run_optimization(self, population_size=100, n_generations=50):
        """运行遗传优化"""
        pop = self.toolbox.population(n=population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # 运行遗传算法
        pop, log = algorithms.eaSimple(
            pop, 
            self.toolbox, 
            cxpb=0.7,  # 交叉概率
            mutpb=0.3,  # 突变概率
            ngen=n_generations, 
            stats=stats,
            halloffame=hof,
            verbose=True
        )
        
        return pop, log, hof

def generate_mock_data(n_days=1000, n_coins=10):
    """
    生成模拟的加密货币数据
    
    参数:
    - n_days: 天数
    - n_coins: 币种数量
    
    返回:
    - dict: 包含OHLCV数据的字典
    """
    np.random.seed(42)
    dates = pd.date_range(end='2024-01-01', periods=n_days, freq='D')
    coins = [f'COIN_{i}' for i in range(n_coins)]
    
    # 生成基础价格序列
    base_prices = {}
    for coin in coins:
        # 生成随机游走价格
        price = 100 * np.exp(np.random.randn(n_days) * 0.02).cumprod()
        base_prices[coin] = price
    
    # 创建OHLCV数据
    data_dict = {
        'open': pd.DataFrame(index=dates, columns=coins),
        'high': pd.DataFrame(index=dates, columns=coins),
        'low': pd.DataFrame(index=dates, columns=coins),
        'close': pd.DataFrame(index=dates, columns=coins),
        'volume': pd.DataFrame(index=dates, columns=coins)
    }
    
    for coin in coins:
        base_price = base_prices[coin]
        
        # 生成OHLC数据
        data_dict['open'][coin] = base_price * (1 + np.random.randn(n_days) * 0.01)
        data_dict['high'][coin] = base_price * (1 + np.abs(np.random.randn(n_days) * 0.02))
        data_dict['low'][coin] = base_price * (1 - np.abs(np.random.randn(n_days) * 0.02))
        data_dict['close'][coin] = base_price * (1 + np.random.randn(n_days) * 0.01)
        
        # 生成成交量数据
        data_dict['volume'][coin] = np.random.lognormal(10, 1, n_days)
    
    return data_dict

def run_demo():
    """运行因子挖掘演示"""
    # 1. 生成模拟数据
    print("生成模拟数据...")
    data_dict = generate_mock_data(n_days=1000, n_coins=10)
    
    # 2. 初始化因子挖掘器
    print("\n初始化因子挖掘器...")
    miner = FactorMiner(data_dict)
    
    # 3. 运行优化
    print("\n开始因子优化...")
    pop, log, hof = miner.run_optimization(
        population_size=50,  # 使用较小的种群以加快演示
        n_generations=20     # 使用较少的代数以加快演示
    )
     
    # 4. 分析结果
    print("\n分析优化结果...")
    best_factor = hof[0]
    print("\n最优因子表达式:")
    print(best_factor)
    
    best_ic = miner.evaluate_factor(best_factor)[0]
    print(f"\n最优因子IC值: {best_ic:.4f}")
    
    # 5. 可视化优化过程
    print("\n绘制优化过程...")
    import matplotlib.pyplot as plt
    
    gen = log.select("gen")
    fit_mins = log.select("min")
    fit_avgs = log.select("avg")
    fit_maxs = log.select("max")
    
    plt.figure(figsize=(10, 6))
    plt.plot(gen, fit_mins, "b-", label="Minimum Fitness")
    plt.plot(gen, fit_avgs, "r-", label="Average Fitness")
    plt.plot(gen, fit_maxs, "g-", label="Maximum Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (IC)")
    plt.title("Evolution of Factor IC over Generations")
    plt.legend(loc="best")
    plt.grid(True)
    
    # 6. 计算并展示最优因子的值
    print("\n计算最优因子值...")
    func = miner.toolbox.compile(expr=best_factor)
    factor_value = func(
        data_dict['open'],
        data_dict['high'],
        data_dict['low'],
        data_dict['close'],
        data_dict['volume']
    )
    
    # 展示因子值的统计特征
    factor_df = pd.DataFrame(factor_value)
    print("\n因子值统计特征:")
    print(factor_df.describe())
    
    # 7. 绘制因子值热力图
    print("\n绘制因子值热力图...")
    plt.figure(figsize=(12, 8))
    plt.imshow(factor_df.iloc[-50:].T, aspect='auto', cmap='RdYlBu')
    plt.colorbar(label='Factor Value')
    plt.title('Factor Values Heatmap (Last 50 Days)')
    plt.xlabel('Time')
    plt.ylabel('Coins')
    plt.tight_layout()
    
    plt.show()

if __name__ == '__main__':
    run_demo()