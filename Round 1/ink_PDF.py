import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import quad

# 读取三天数据
day = -2
market_data = pd.read_csv(f"../dashboard/round-1-island-data-bottle/prices_round_1_day_{day}.csv", sep=";", header=0)

day = -1
market_data2 = pd.read_csv(f"../dashboard/round-1-island-data-bottle/prices_round_1_day_{day}.csv", sep=";", header=0)

day = 0
market_data3 = pd.read_csv(f"../dashboard/round-1-island-data-bottle/prices_round_1_day_{day}.csv", sep=";", header=0)

# 合并三天数据
combined_data = pd.concat([market_data, market_data2, market_data3])

# 提取 SQUID_INK 数据
ink_data = combined_data[combined_data['product'] == 'SQUID_INK']
ink_prices = ink_data['mid_price']


# 拟合 KDE
kde = gaussian_kde(ink_prices)

# 计算 P(X < 1900)
prob_less_than_1900, _ = quad(kde, -np.inf, 1900)

# 计算 P(X > 2029)
prob_greater_than_2029, _ = quad(kde, 2029, np.inf)

print(f"P(mid_price < 1900) = {prob_less_than_1900:.4f} ({prob_less_than_1900*100:.2f}%)")
print(f"P(mid_price > 2029) = {prob_greater_than_2029:.4f} ({prob_greater_than_2029*100:.2f}%)")

# 绘制密度函数图
plt.figure(figsize=(10, 6))
sns.kdeplot(ink_prices, shade=True, color='blue')
plt.title('Density Plot of SQUID_INK Mid Prices (3 Days Combined)')
plt.xlabel('Mid Price')
plt.ylabel('Density')
plt.grid(True)
plt.show()


