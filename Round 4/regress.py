import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

day = 1
market_data1 = pd.read_csv(
    f"../dashboard/round-4-island-data-bottle/prices_round_4_day_{day}.csv",
    sep=";",
    header=0,
)

day = 2
market_data2 = pd.read_csv(
    f"../dashboard/round-4-island-data-bottle/prices_round_4_day_{day}.csv",
    sep=";",
    header=0,
)

day = 3
market_data3 = pd.read_csv(
    f"../dashboard/round-4-island-data-bottle/prices_round_4_day_{day}.csv",
    sep=";",
    header=0,
)

day = 1
observations_data1 = pd.read_csv(
    f"../dashboard/round-4-island-data-bottle/observations_round_4_day_{day}.csv",

    header=0,
)

day = 2
observations_data2 = pd.read_csv(
    f"../dashboard/round-4-island-data-bottle/observations_round_4_day_{day}.csv",

    header=0,
)

day = 3
observations_data3 = pd.read_csv(
    f"../dashboard/round-4-island-data-bottle/observations_round_4_day_{day}.csv",

    header=0,
)



# 合并三天数据
market_data = pd.concat([market_data1, market_data2, market_data3])
observations_data = pd.concat([observations_data1,observations_data2,observations_data3])

MACARONS_data = market_data[market_data["product"] == "MAGNIFICENT_MACARONS"]
MACARONS_prices = MACARONS_data["mid_price"]
SUGAR_prices = observations_data['sugarPrice']
SUNLIGHT_index = observations_data['sunlightIndex']

# 合并数据
merged_data = MACARONS_data[['timestamp', 'mid_price']].merge(
    observations_data[['timestamp', 'sugarPrice', 'sunlightIndex']],
    on='timestamp',
    how='inner'
)
# 按时间戳排序
merged_data = merged_data.sort_values('timestamp')
# 计算相关系数矩阵
correlation_matrix = merged_data[['mid_price', 'sugarPrice', 'sunlightIndex']].corr()
print("相关系数矩阵:\n", correlation_matrix)

import statsmodels.api as sm
# 使用中介效应工具（如Mediation包）或逐步回归
# Step 1: 阳光指数 -> 糖价
model1 = sm.OLS(merged_data['sugarPrice'], sm.add_constant(merged_data['sunlightIndex'])).fit()
# Step 2: 阳光指数 + 糖价 -> 马卡龙价格
model2 = sm.OLS(merged_data['mid_price'], sm.add_constant(merged_data[['sunlightIndex', 'sugarPrice']])).fit()
# 计算中介效应比例
indirect = model1.params['sunlightIndex'] * model2.params['sugarPrice']
direct = model2.params['sunlightIndex']
total_effect = indirect + direct
print(total_effect)
print(indirect)
print(direct)
print(model1.summary())  # 检查阳光对糖价的显著性
print(model2.summary())  # 检查糖价对马卡龙的显著性