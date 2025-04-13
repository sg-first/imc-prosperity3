import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from matplotlib import rcParams



# 定义要读取的天数
days = [-1, 0, 1]

# 读取所有天数的数据并合并
all_market_data = pd.concat([
    pd.read_csv(f"round-2-island-data-bottle/prices_round_2_day_{day}.csv", sep=";", header=0).assign(day=day)
    for day in days
])

# 筛选各个产品的数据
djembes = all_market_data[all_market_data['product'] == 'DJEMBES']
croissants = all_market_data[all_market_data['product'] == 'CROISSANTS']
jams = all_market_data[all_market_data['product'] == 'JAMS']
picnic_basket1 = all_market_data[all_market_data['product'] == 'PICNIC_BASKET1']
picnic_basket2 = all_market_data[all_market_data['product'] == 'PICNIC_BASKET2']

# 合并数据，确保每个时间步都有对应的价格
merged_data = pd.merge(picnic_basket1, croissants, on=['timestamp', 'day'], suffixes=('_basket1', '_croissants'))
merged_data = pd.merge(merged_data, jams, on=['timestamp', 'day'], suffixes=('', '_jams'))
merged_data = pd.merge(merged_data, djembes, on=['timestamp', 'day'], suffixes=('', '_djembes'))
merged_data = pd.merge(merged_data, picnic_basket2, on=['timestamp', 'day'], suffixes=('', '_basket2'))

# 重命名列以避免冲突
merged_data.rename(columns={
    'mid_price_basket1': 'mid_price_picnic_basket1',
    'mid_price_croissants': 'mid_price_croissants',
    'mid_price': 'mid_price_jams',
    'mid_price_djembes': 'mid_price_djembes',
    'mid_price_basket2': 'mid_price_picnic_basket2'
}, inplace=True)

# 计算每个时间步的价差
# PICNIC_BASKET1的价差：PICNIC_BASKET1价格 - (6*CROISSANTS + 3*JAMS + 1*DJEMBES)
merged_data['spread1'] = merged_data['mid_price_picnic_basket1'] - (merged_data['mid_price_croissants'] * 6 + merged_data['mid_price_jams'] * 3 + merged_data['mid_price_djembes'])

# PICNIC_BASKET2的价差：PICNIC_BASKET2价格 - (4*CROISSANTS + 2*JAMS)
merged_data['spread2'] = merged_data['mid_price_picnic_basket2'] - (merged_data['mid_price_croissants'] * 4 + merged_data['mid_price_jams'] * 2)

def check_stationarity(spread_series):
    """ADF平稳性检验"""
    result = adfuller(spread_series.dropna())
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    if result[1] < 0.05:
        print("价差序列平稳，可能具有均值回归特性")
    else:
        print("价差序列非平稳，需进一步分析")


def hurst_exponent(spread_series):
    """改进的Hurst指数计算"""
    spread_series = spread_series.dropna()
    if len(spread_series) < 100:
        return np.nan

    lags = range(10, min(100, len(spread_series) // 2))  # 调整滞后范围
    tau = []
    for lag in lags:
        if lag == 0 or 2 * lag > len(spread_series):
            continue
        # 添加微小常数避免零值
        diff = np.std(np.subtract(spread_series[lag:], spread_series[:-lag]) + 1e-8)
        tau.append(diff)

        if len(tau) < 2:
            return np.nan

    poly = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
    return poly[0] * 2



# 3. 半衰期计算
def calculate_half_life(spread_series):
    spread_series = spread_series.dropna()
    if len(spread_series) < 3:
        return np.nan

    spread_lag = spread_series.shift(1).dropna()
    delta_spread = spread_series.diff().dropna()

    # 确保长度一致
    min_len = min(len(spread_lag), len(delta_spread))
    spread_lag = spread_lag.iloc[:min_len]
    delta_spread = delta_spread.iloc[:min_len]

    X = sm.add_constant(spread_lag)
    model = sm.OLS(delta_spread, X).fit()

    if model.params[1] >= 0:
        return np.inf  # 无效结果
    return -np.log(2) / model.params[1]

# 4. 可视化
rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体
rcParams['axes.unicode_minus'] = False    # 解决负号显示问题


def plot_spread_analysis(spread_series):
    plt.figure(figsize=(12, 6))

    # 计算关键指标
    mean_val = spread_series.mean()
    std_val = spread_series.std()
    half_life = calculate_half_life(spread_series)

    # 绘制价差序列
    plt.plot(spread_series.index, spread_series,
             label=f'价差序列 (Hurst={hurst_exponent(spread_series):.2f}',
             color='#1f77b4',
             alpha=0.7)

    # 绘制均值线
    plt.axhline(mean_val, color='#d62728', linestyle='--',
                label=f'均值 ({mean_val:.2f})')

    # 绘制标准差区间
    plt.fill_between(spread_series.index,
                     mean_val - std_val,
                     mean_val + std_val,
                     color='gray', alpha=0.2,
                     label='±1标准差区间')

    # 标注关键区域
    plt.text(0.05, 0.95,
             f"半衰期: {half_life:.1f}天\n标准差: {std_val:.2f}",
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))

    # 图形美化
    plt.title("野餐篮2价差均值回归分析", fontsize=14)
    plt.xlabel("时间序列", fontsize=12)
    plt.ylabel("价差", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# 1. ADF检验
check_stationarity(merged_data['spread2'].dropna())

# 2. Hurst指数
H = hurst_exponent(merged_data['spread2'].dropna())
print(f"Hurst指数: {H:.2f}")
# 3.半衰期
a = calculate_half_life(merged_data['spread2'])
print('半衰期：',a)
plot_spread_analysis(merged_data['spread2'] )
