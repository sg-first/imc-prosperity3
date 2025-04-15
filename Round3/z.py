import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton

rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

day = 0
market_data = pd.read_csv(
    f"round-3-island-data-bottle/prices_round_3_day_{day}.csv",
    sep=";", header=0,
)

day = 1
market_data1 = pd.read_csv(
    f"round-3-island-data-bottle/prices_round_3_day_{day}.csv",
    sep=";", header=0,
)

day = 2
market_data2 = pd.read_csv(
    f"round-3-island-data-bottle/prices_round_3_day_{day}.csv",
    sep=";", header=0,
)

# 合并三天数据
combined_data = pd.concat([market_data, market_data1, market_data2])

VOLCANIC_ROCK_VOUCHER_9500_data = combined_data[combined_data["product"] == "VOLCANIC_ROCK_VOUCHER_9500"]
VOLCANIC_ROCK_VOUCHER_9500 = VOLCANIC_ROCK_VOUCHER_9500_data["mid_price"]

VOLCANIC_ROCK_data = combined_data[combined_data["product"] == "VOLCANIC_ROCK"]
VOLCANIC_ROCK = VOLCANIC_ROCK_data['mid_price']

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
r = 0                 # 假设无风险利率为2%（需根据实际情况修改）

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
VOLCANIC_ROCK_VOUCHER_9500_data.loc[:, 'implied_vol'] = iv_results  # 使用.loc[]避免SettingWithCopyWarning

# 输出结果:时间戳+期权中间价+隐含波动率
print(VOLCANIC_ROCK_VOUCHER_9500_data[['timestamp', 'mid_price', 'implied_vol']])


# 确保volatility_df是DataFrame
volatility_df = pd.DataFrame()  # 创建一个空的DataFrame
volatility_df['Volatility'] = VOLCANIC_ROCK_VOUCHER_9500_data['implied_vol'].dropna()

# 计算波动率的滚动均值和标准差（用于计算Z-score）
volatility_df['Rolling Mean'] = volatility_df['Volatility'].rolling(window=30).mean()
volatility_df['Rolling Std'] = volatility_df['Volatility'].rolling(window=30).std()

# 计算Z-score
volatility_df['Z-score'] = (volatility_df['Volatility'] - volatility_df['Rolling Mean']) / volatility_df['Rolling Std']

# 设置不同的zscore_threshold来进行回测

zscore_thresholds = np.arange(1, 100.1, 0.1)
results = {}

# 假设初始资金为10000，模拟交易
initial_cash = 10000
cash = initial_cash
position = 0  # 初始仓位
trade_log = []


# 回测函数
def backtest_strategy(zscore_threshold, volatility_df):
    cash = initial_cash
    position = 0
    pnl = []
    for i in range(30, len(volatility_df)):  # 从第30天开始，因为我们需要前30天的均值和标准差
        current_row = volatility_df.iloc[i]
        zscore = current_row['Z-score']

        # 确保 zscore 是标量值
        if np.isnan(zscore):
            continue  # 跳过无效的Z-score

        if zscore >= zscore_threshold:  # 卖出信号
            if position > 0:
                cash += position * current_row['Volatility']  # 假设价格就是波动率
                position = 0  # 卖出所有持仓
                trade_log.append(('SELL', current_row.name, current_row['Volatility'], position))
        elif zscore <= -zscore_threshold:  # 买入信号
            if position == 0:
                position = cash // current_row['Volatility']  # 用全部资金购买
                cash -= position * current_row['Volatility']  # 扣除资金
                trade_log.append(('BUY', current_row.name, current_row['Volatility'], position))

        pnl.append(cash + position * current_row['Volatility'])  # 计算每日的现金加仓位价值

    # 计算回测结果
    total_pnl = pnl[-1] - initial_cash  # 最终资产减去初始现金即为总利润
    return total_pnl, pnl


# 回测不同的阈值并比较
for zscore_threshold in zscore_thresholds:
    total_pnl, pnl = backtest_strategy(zscore_threshold, volatility_df)
    results[zscore_threshold] = {
        'Total PnL': total_pnl,
        'PnL History': pnl
    }

# 选择最优的zscore_threshold（根据总利润来选择最优）
best_threshold = max(results, key=lambda x: results[x]['Total PnL'])

# 打印最优结果
print(f"最优的 Z-score 阈值是: {best_threshold}")
print(f"总利润: {results[best_threshold]['Total PnL']}")

# 绘制不同阈值下的收益曲线
plt.figure(figsize=(10, 6))
for zscore_threshold in zscore_thresholds:
    plt.plot(results[zscore_threshold]['PnL History'], label=f'Threshold {zscore_threshold}')
plt.title('不同 Z-score 阈值下的回测收益')
plt.xlabel('日期')
plt.ylabel('资产价值')
plt.legend()
plt.show()
