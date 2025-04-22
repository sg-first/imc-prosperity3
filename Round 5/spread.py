import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


day = 1
market_data2 = pd.read_csv(
    f"../dashboard/round-4-island-data-bottle/prices_round_4_day_{day}.csv",
    sep=";",
    header=0,
)

day = 2
market_data3 = pd.read_csv(
    f"../dashboard/round-5-island-data-bottle/prices_round_5_day_{day}.csv",
    sep=";",
    header=0,
)

day = 3
market_data4 = pd.read_csv(
    f"../dashboard/round-5-island-data-bottle/prices_round_5_day_{day}.csv",
    sep=";",
    header=0,
)

day = 4
market_data5 = pd.read_csv(
    f"../dashboard/round-5-island-data-bottle/prices_round_5_day_{day}.csv",
    sep=";",
    header=0,
)


day = 1
observations_data2 = pd.read_csv(
    f"../dashboard/round-4-island-data-bottle/observations_round_4_day_{day}.csv",

    header=0,
)

day = 2
observations_data3 = pd.read_csv(
    f"../dashboard/round-5-island-data-bottle/observations_round_5_day_{day}.csv",

    header=0,
)

day = 3
observations_data4 = pd.read_csv(
    f"../dashboard/round-5-island-data-bottle/observations_round_5_day_{day}.csv",

    header=0,
)

day = 4
observations_data5 = pd.read_csv(
    f"../dashboard/round-5-island-data-bottle/observations_round_5_day_{day}.csv",

    header=0,
)



combined_data1 = pd.concat([market_data2,market_data3,market_data4,market_data5])
MACARONS_data = combined_data1[combined_data1["product"] == "MAGNIFICENT_MACARONS"]
MACARONS_prices = MACARONS_data["mid_price"]



combined_data2 = pd.concat([observations_data2,observations_data3,observations_data4,observations_data5])

bid_prices = combined_data2["bidPrice"]
ask_prices = combined_data2["askPrice"]
combined_data2["mid_price"] =(combined_data2["bidPrice"] + combined_data2["askPrice"])/2

aa = combined_data2['mid_price'].reset_index(drop=True)
bb = MACARONS_prices.reset_index(drop=True)
c = aa - bb

print(c)
plt.scatter(range(len(aa)),c,s = 1)
plt.show()

