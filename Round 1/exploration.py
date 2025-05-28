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

day = 0
market_data = pd.read_csv(f"../dashboard/round-1-island-data-bottle/prices_round_1_day_{day}.csv", sep=";", header=0)
trade_history = pd.read_csv(f"../dashboard/round-1-island-data-bottle/trades_round_1_day_{day}.csv", sep=";", header=0)

kelp = market_data[market_data['product'] == 'KELP']
kelp_price = kelp['mid_price']
#print(len(kelp_price))

ink = market_data[market_data['product'] == 'SQUID_INK']
ink_price = ink['mid_price']
#print(len(ink_price))

ink_diff1 = ink_price.diff(periods=1).dropna().values
kelp_diff1 = kelp_price.diff(periods=1).dropna().values
# 平稳性检验,已经特么的通过了
from statsmodels.tsa.stattools import adfuller
#print("海带价格平稳性p值:", adfuller(kelp_diff1)[1])
#print("墨汁价格平稳性p值:", adfuller(ink_diff1)[1])
data = np.column_stack((ink_diff1,kelp_diff1))
print(len(ink_diff1))
print(len(kelp_diff1))

#
df = [ink_price,ink_diff1]

model = VAR(data)
lag_order = model.select_order()  # 自动选择最优滞后
print("Optimal lag order:", lag_order.selected_orders['aic'])
results = model.fit(lag_order.selected_orders['aic'])
print(results.summary())


# # 格兰杰检验,检查p值是否<0.05
# granger_test = grangercausalitytests(data, maxlag=10)
#
# # 标准化数据（消除量纲影响）
# kelp_prices = (kelp_diff1 - kelp_diff1.mean()) / kelp_diff1.std()
# ink_prices = (ink_diff1 - ink_diff1.mean()) / ink_diff1.std()
#
# # 计算CCF（最大滞后20步）
# max_lag = 15
# ccf_values = ccf(ink_prices, kelp_prices, adjusted=False)[:max_lag + 1]
# lags = np.arange(0, max_lag + 1)
#
# # 找到最强相关的滞后阶数
# optimal_lag = lags[np.argmax(np.abs(ccf_values))]
# print(f"最优滞后阶数: {optimal_lag}, 相关系数: {ccf_values[optimal_lag]:.3f}")
#
# plt.stem(lags, ccf_values, linefmt='b-', markerfmt='bo', basefmt='k-')
# plt.axhline(0, color='k', linestyle='--')
# plt.title("Kelp vs Squid Ink 交叉相关性")
# plt.xlabel("滞后阶数 (Lag)")
# plt.ylabel("相关系数")
# plt.grid(True)
# plt.show()









# 使用滞后4-7阶的KELP系数
kelp_coeffs = np.array([
    -0.18454702, -0.13158352, -0.15589962, -0.11816887   # L7.y2
])

# 生成预测值 (需要至少7期数据才能获取L4-L7)
predictions = []
for i in range(7, len(kelp_diff1)):
    window = kelp_diff1[i-4:i][::-1]  # 获取L4-L7的差分价格(逆序)
    pred = np.dot(window, kelp_coeffs)
    predictions.append(pred)

# 对齐实际值
actual = ink_diff1[7:]

# 确保所有数组长度一致
min_length = min(len(predictions), len(actual))
predictions = predictions[:min_length]
actual = actual[:min_length]

# 创建详细结果DataFrame
result_df = pd.DataFrame({
    'time_step': range(7, 7 + min_length),
    'kelp_L4': kelp_diff1[3:3+min_length],  # L4数据
    'kelp_L5': kelp_diff1[2:2+min_length],  # L5数据
    'kelp_L6': kelp_diff1[1:1+min_length],  # L6数据
    'kelp_L7': kelp_diff1[:min_length],     # L7数据
    'predicted': predictions,
    'actual': actual,
    'sign_match': np.sign(predictions) == np.sign(actual)
})

# 符号一致性检验
sign_match = result_df['sign_match']
match_rate = np.mean(sign_match)

# 输出核心结果
print(f"\n符号检验结果汇总:")
print(f"总样本数: {len(sign_match)}")
print(f"符号一致比例: {match_rate:.2%}")
print(f"预测均值: {np.mean(predictions):.6f}")
print(f"实际均值: {np.mean(actual):.6f}")

# 输出前20期详细结果
print("\n前20期详细预测情况:")
print(result_df.head(20).to_string(float_format="%.6f"))