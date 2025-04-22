import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import quad

day = -2
market_data0 = pd.read_csv(
    f"../dashboard/round-1-island-data-bottle/prices_round_1_day_{day}.csv",
    sep=";",
    header=0,
)

day = -1
market_data = pd.read_csv(
    f"../dashboard/round-2-island-data-bottle/prices_round_2_day_{day}.csv",
    sep=";",
    header=0,
)

day = 0
market_data1 = pd.read_csv(
    f"../dashboard/round-2-island-data-bottle/prices_round_2_day_{day}.csv",
    sep=";",
    header=0,
)

day = 1
market_data2 = pd.read_csv(
    f"../dashboard/round-2-island-data-bottle/prices_round_2_day_{day}.csv",
    sep=";",
    header=0,
)

day = 2
market_data3 = pd.read_csv(
    f"../dashboard/round-3-island-data-bottle/prices_round_3_day_{day}.csv",
    sep=";",
    header=0,
)

day = 3
market_data4 = pd.read_csv(
    f"../dashboard/round-4-island-data-bottle/prices_round_4_day_{day}.csv",
    sep=";",
    header=0,
)

day = 4
market_data5 = pd.read_csv(
    f"../dashboard/round-5-island-data-bottle/prices_round_5_day_{day}.csv",
    sep=";",
    header=0,
)



# 合并三天数据
combined_data = pd.concat([market_data0, market_data, market_data1, market_data2,market_data3,market_data4,market_data5])

ink_data = combined_data[combined_data["product"] == "VOLCANIC_ROCK"]
ink_prices = ink_data["mid_price"]
print(len(ink_prices))


# 拟合 KDE
kde = gaussian_kde(ink_prices)

# 计算 P(X < 1900)
prob_less_than_1900, _ = quad(kde, -np.inf, 1781)

# 计算 P(X > 2029)
prob_greater_than_2029, _ = quad(kde, 2029, np.inf)

print(f"P(mid_price < 1781) = {prob_less_than_1900:.4f} ({prob_less_than_1900*100:.2f}%)")
print(f"P(mid_price > 2029) = {prob_greater_than_2029:.4f} ({prob_greater_than_2029*100:.2f}%)")

# 绘制密度函数图
plt.figure(figsize=(10, 6))
sns.kdeplot(ink_prices, shade=True, color='blue')
plt.title('Density Plot of SQUID_INK Mid Prices (3 Days Combined)')
plt.xlabel('Mid Price')
plt.ylabel('Density')
plt.grid(True)
plt.show()