import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.tsa.stattools import ccf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.vector_ar.var_model import VAR

day = -2
market_data = pd.read_csv(f"../dashboard/round-1-island-data-bottle/prices_round_1_day_{day}.csv", sep=";", header=0)
trade_history = pd.read_csv(f"../dashboard/round-1-island-data-bottle/trades_round_1_day_{day}.csv", sep=";", header=0)

kelp = market_data[market_data['product'] == 'KELP']
kelp_price = kelp['mid_price']
#print(len(kelp_price))

ink = market_data[market_data['product'] == 'SQUID_INK']
ink_price = ink['mid_price']
#print(len(ink_price))

ink_diff1 = ink_price.diff(periods=1).dropna()
kelp_diff1 = kelp_price.diff(periods=1).dropna()
# 平稳性检验,已经特么的通过了
from statsmodels.tsa.stattools import adfuller
#print("海带价格平稳性p值:", adfuller(kelp_diff1)[1])
#print("墨汁价格平稳性p值:", adfuller(ink_diff1)[1])
data = np.column_stack((ink_diff1,kelp_diff1))
print(len(ink_diff1))
print(len(kelp_diff1))


model = VAR(data)
lag_order = model.select_order()  # 自动选择最优滞后
print("Optimal lag order:", lag_order.selected_orders['aic'])
# 格兰杰检验,检查p值是否<0.05
granger_test = grangercausalitytests(data, maxlag=10)


# 标准化数据（消除量纲影响）
kelp_prices = (kelp_diff1 - kelp_diff1.mean()) / kelp_diff1.std()
ink_prices = (ink_diff1 - ink_diff1.mean()) / ink_diff1.std()

# 计算CCF（最大滞后20步）
max_lag = 15
ccf_values = ccf(ink_prices, kelp_prices, adjusted=False)[:max_lag + 1]
lags = np.arange(0, max_lag + 1)

# 找到最强相关的滞后阶数
optimal_lag = lags[np.argmax(np.abs(ccf_values))]
print(f"最优滞后阶数: {optimal_lag}, 相关系数: {ccf_values[optimal_lag]:.3f}")

plt.stem(lags, ccf_values, linefmt='b-', markerfmt='bo', basefmt='k-')
plt.axhline(0, color='k', linestyle='--')
plt.title("Kelp vs Squid Ink 交叉相关性")
plt.xlabel("滞后阶数 (Lag)")
plt.ylabel("相关系数")
plt.grid(True)
plt.show()