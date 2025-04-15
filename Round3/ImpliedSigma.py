import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

import numpy as np

from scipy.stats import norm
from scipy.optimize import newton
from MeanReversion import MeanReversionAnalyzer


rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


day = 0
market_data = pd.read_csv(
    f"round-3-island-data-bottle/prices_round_3_day_{day}.csv",
    sep=";",
    header=0,
)

day = 1
market_data1 = pd.read_csv(
    f"round-3-island-data-bottle/prices_round_3_day_{day}.csv",
    sep=";",
    header=0,
)

day = 2
market_data2 = pd.read_csv(
    f"round-3-island-data-bottle/prices_round_3_day_{day}.csv",
    sep=";",
    header=0,
)

# 合并三天数据
combined_data = pd.concat([ market_data, market_data1, market_data2])

VOLCANIC_ROCK_VOUCHER_9500_data = combined_data[combined_data["product"] == "VOLCANIC_ROCK_VOUCHER_9500"]
VOLCANIC_ROCK_VOUCHER_9500 = VOLCANIC_ROCK_VOUCHER_9500_data["mid_price"]
# print(VOLCANIC_ROCK_VOUCHER_9500)

VOLCANIC_ROCK_data = combined_data[combined_data["product"] == "VOLCANIC_ROCK"]
VOLCANIC_ROCK = VOLCANIC_ROCK_data['mid_price']
# print(VOLCANIC_ROCK)

# Black-Scholes看涨期权定价公式
def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# 计算隐含波动率（牛顿迭代法）
def implied_volatility(C_market, S, K, T, r, initial_guess=0.2):
    def objective(sigma):
        return bs_call(S, K, T, r, sigma) - C_market
    try:
        iv = newton(objective, initial_guess, maxiter=100)
        return iv
    except RuntimeError:
        return np.nan  # 处理无法收敛的情况

# ----------------------------------
# 根据你的数据输入参数
# ----------------------------------
# 假设参数：
K = 9500              # 行权价（已知）
T = 5 / 252           # 剩余到期时间（按自然日计算年化）
r = 0              # 假设无风险利率为2%（需根据实际情况修改）

# 从数据中提取标的价格和期权价格
S = VOLCANIC_ROCK.values            # 标的物价格（火山岩的mid_price）
C_market = VOLCANIC_ROCK_VOUCHER_9500.values  # 期权市场价格（火山岩券的mid_price）

# 确保数据对齐且长度一致
assert len(S) == len(C_market), "标的物价格与期权价格数据长度不一致"

# 计算隐含波动率
iv_results = []
for s, c in zip(S, C_market):
    if s > 0 and c > 0:  # 过滤无效价格
        iv = implied_volatility(C_market=c, S=s, K=K, T=T, r=r)
        iv_results.append(iv)
    else:
        iv_results.append(np.nan)

# 将结果添加到原始数据中
VOLCANIC_ROCK_VOUCHER_9500_data['implied_vol'] = iv_results

# 输出结果:时间戳+期权中间价+隐含波动率
print(VOLCANIC_ROCK_VOUCHER_9500_data[['timestamp', 'mid_price', 'implied_vol']])

analyzer = MeanReversionAnalyzer(VOLCANIC_ROCK_VOUCHER_9500_data['implied_vol'].dropna())
analyzer.check_stationarity()
analyzer.plot_analysis()