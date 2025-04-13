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


class EWMACalculator:
    def __init__(self, halflife=695):
        self.halflife = halflife
        self.lambda_ = 1 - np.exp(np.log(0.5) / halflife)
        self.reset()

    def reset(self):
        self.mean = None
        self.var = None
        self.last_timestamp = None

    def update(self, value, timestamp):


        # 初始化第一个数据点
        if self.mean is None:
            self.mean = value
            self.var = 0.0
            return 0.0  # 初始Z值为0

        # 计算新的均值和方差
        new_mean = (1 - self.lambda_) * self.mean + self.lambda_ * value
        new_var = (1 - self.lambda_) * self.var + self.lambda_ * (value - new_mean) ** 2

        # 更新状态
        self.mean = new_mean
        self.var = new_var

        # 计算Z值
        return (value - new_mean) / np.sqrt(new_var) if new_var > 1e-8 else 0.0


# 初始化计算器
ewma = EWMACalculator(halflife=695)

# 按时间顺序处理数据
merged_data = merged_data.sort_values(['day', 'timestamp']).reset_index(drop=True)

# 计算Z值
merged_data['z_scores'] = np.nan
for idx, row in merged_data.iterrows():
    merged_data.at[idx, 'z_scores'] = ewma.update(row['spread1'], row['timestamp'])

# 分析阈值
threshold_analysis = pd.DataFrame({
    'threshold': np.arange(1.0, 3.5, 0.1),
    'hit_counts': [np.sum(np.abs(merged_data['z_scores']) > t) for t in np.arange(1.0, 3.5, 0.1)],
    'max_duration': [merged_data['z_scores'].abs().gt(t).astype(int).groupby(
        (~merged_data['z_scores'].abs().gt(t)).cumsum()).cumcount().max()
                     for t in np.arange(1.0, 3.5, 0.1)]
})

# 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(merged_data['timestamp'], merged_data['z_scores'], label='Z-Scores')
plt.axhline(2.0, color='r', linestyle='--', alpha=0.5, label='2σ Threshold')
plt.axhline(-2.0, color='r', linestyle='--', alpha=0.5)
plt.title(f'Spread1 Z-Scores (λ={ewma.lambda_:.4f}, Half-life={ewma.halflife})')
plt.xlabel('Timestamp')
plt.ylabel('Z-Score')
plt.legend()
plt.show()

# 显示阈值分析结果
print("\n阈值分析报告：")
print(threshold_analysis)

# 统计特征分析
print("\nZ值统计特征：")
print(merged_data['z_scores'].describe())