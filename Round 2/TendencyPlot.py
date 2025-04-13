import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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

# 合并三天数据
combined_data = pd.concat([market_data0, market_data, market_data1, market_data2])

ink_data = combined_data[combined_data["product"] == "SQUID_INK"]
ink_prices = ink_data["mid_price"]
print(len(ink_prices))

kelp_dta = combined_data[combined_data["product"] == "KELP"]
kelp_prices = kelp_dta["mid_price"]


plt.plot(
    range(len(ink_prices)), ink_prices, label="ink_price", linewidth=0.5, color="blue"
)
plt.plot(
    range(len(ink_prices)),
    kelp_prices,
    label="kelp_price",
    linewidth=0.5,
    color="green",
    linestyle="--",
)
# 设置图表标题和坐标轴标签
plt.title("Price Tendency", fontsize=14)
plt.xlabel("time", fontsize=12)
plt.ylabel("price", fontsize=12)
plt.legend()

# 旋转x轴标签以提高可读性
plt.xticks(rotation=45)

# 显示网格线
plt.grid(True)

# 调整布局以避免标签重叠
plt.tight_layout()


# 显示图表

plt.show()
