import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from matplotlib import rcParams
import seaborn as sns

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
merged_data['merge1'] = (merged_data['mid_price_croissants'] * 0.6 + merged_data['mid_price_jams'] * 0.3 + merged_data['mid_price_djembes']*0.1)

# PICNIC_BASKET2的价差：PICNIC_BASKET2价格 - (4*CROISSANTS + 2*JAMS)
merged_data['merge2'] = (merged_data['mid_price_croissants'] * 4/6 + merged_data['mid_price_jams'] * 2/6)

# print(merged_data['merge2'])
# print(merged_data['mid_price_picnic_basket2'])
plt.plot(range(len(merged_data['merge2'])),merged_data['merge2'],color = 'blue')
#sns.kdeplot(merged_data['merge1'], shade=True, color='blue')
plt.plot(range(len(merged_data['merge2'])),merged_data['mid_price_picnic_basket2']/6,color = 'green')
plt.show()
a = merged_data['merge1'].diff().std()









merged_data['a'] = merged_data['merge2'].diff()
merged_data['b'] = merged_data['mid_price_picnic_basket2'].diff()




# 特征工程
def create_features(df):
    # 计算合成价格 - 使用正确的列名
    df["synthetic"] = 4 * df["mid_price_croissants"] + 2 * df["mid_price_jams"]

    # 重命名basket2列
    df = df.rename(columns={'mid_price_picnic_basket2': 'basket2'})

    # 计算差分序列（使用百分比变化避免量纲问题）
    df["basket2_diff"] = df["basket2"].pct_change() * 100  # 百分比变化
    df["synthetic_diff"] = df["synthetic"].pct_change() * 100

    # 创建滞后特征
    lags = [1, 2, 3, 4]
    for lag in lags:
        df[f"synthetic_lag{lag}"] = df["synthetic_diff"].shift(lag)

    return df

# 模型预测
def generate_predictions(df, coefficients):
    # 使用滞后特征和系数生成预测值
    df["pred_diff"] = 0
    for lag, coef in coefficients.items():
        df["pred_diff"] += coef * df[f"synthetic_lag{lag}"]

    # 生成符号预测
    df["pred_sign"] = np.sign(df["pred_diff"])
    df["true_sign"] = np.sign(df["basket2_diff"])

    return df

# 结果分析
def analyze_results(df):
    # 过滤无效数据（前4个时间点没有足够滞后）
    valid_df = df.dropna(subset=["pred_sign", "true_sign"])

    # 计算准确率
    accuracy = (valid_df["pred_sign"] == valid_df["true_sign"]).mean()

    # 详细分析
    print(f"总样本数：{len(valid_df)}")
    print(f"预测准确率：{accuracy:.2%}")
    print("\n符号分布：")
    print(
        pd.crosstab(
            valid_df["pred_sign"],
            valid_df["true_sign"],
            rownames=["Predicted"],
            colnames=["Actual"],
        )
    )

    # 可视化预测效果
    plt.figure(figsize=(12, 6))
    plt.plot(valid_df["timestamp"], valid_df["basket2_diff"], label="Actual", alpha=0.7)
    plt.plot(valid_df["timestamp"], valid_df["pred_diff"], label="Predicted", alpha=0.7)
    plt.title("Basket2价格差分预测效果")
    plt.ylabel("百分比变化")
    plt.legend()
    plt.show()

    return valid_df

# 系数配置
coefficients = {1: 0.452987, 2: 0.174008, 3: 0.028813, 4: 0.054057}

# 执行流程
df = create_features(merged_data)
df = generate_predictions(df, coefficients)
results = analyze_results(df)


# ================= 分析函数 =================
def manual_ccf(x, y, max_lag=20):
    """手动计算交叉相关系数"""
    ccf = []
    x = x - np.mean(x)
    y = y - np.mean(y)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x_shift = x[:lag]
            y_shift = y[-lag:]
        elif lag > 0:
            x_shift = x[lag:]
            y_shift = y[:-lag]
        else:
            x_shift, y_shift = x, y
        corr = np.corrcoef(x_shift, y_shift)[0, 1]
        ccf.append(corr)
    return ccf


def print_granger_results(results_dict):
    """正确格式化打印格兰杰检验结果"""
    for lag in sorted(results_dict.keys()):
        p_value = results_dict[lag][0]['ssr_chi2test'][1]
        print(f"滞后 {lag} 期: p值 = {p_value:.4f}",
              "***" if p_value < 0.01 else
              "**" if p_value < 0.05 else
              "*" if p_value < 0.1 else "")









# ================= 主要分析流程 =================
if __name__ == "__main__":
    # 数据预处理
    spreads = merged_data[['a', 'b']].dropna()

    # 平稳性检验
    print("\n=== 平稳性检验 ===")
    for col in spreads.columns:
        pvalue = adfuller(spreads[col].dropna())[1]
        print(f"{col}的ADF检验p值: {pvalue:.4f}",
              "(平稳)" if pvalue < 0.05 else "(非平稳)")

    # 交叉相关分析
    rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体
    rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    plt.figure(figsize=(12, 6))
    max_lag = 20
    ccf = manual_ccf(spreads['a'], spreads['b'], max_lag)
    lags = np.arange(-max_lag, max_lag + 1)

    plt.stem(lags, ccf, use_line_collection=True)
    plt.axhline(1.96 / np.sqrt(len(spreads)), color='r', ls='--', label='95%置信区间')
    plt.axhline(-1.96 / np.sqrt(len(spreads)), color='r', ls='--')
    plt.title("a与b交叉相关分析")
    plt.xlabel("滞后阶数（正滞后表示a领先）")
    plt.ylabel("相关系数")
    plt.legend()
    plt.show()


    print("\n=== 格兰杰因果检验 ===")
    max_lag = 5

    print("\n检验b是否领先a:")
    granger_spread1 = grangercausalitytests(spreads[['a', 'b']], max_lag, verbose=False)
    print_granger_results(granger_spread1)

    print("\n检验a是否领先b:")
    granger_spread2 = grangercausalitytests(spreads[['b', 'a']], max_lag, verbose=False)
    print_granger_results(granger_spread2)

    # VAR模型分析
    print("\n=== VAR模型分析 ===")
    var_model = VAR(spreads)
    lag_order = var_model.select_order(maxlags=10).aic
    print(f"根据AIC选择的最优滞后阶数: {lag_order}")

    var_result = var_model.fit(lag_order)
    print("\n模型摘要:")
    print(var_result.summary())

    # 脉冲响应分析
    irf = var_result.irf(10)
    irf.plot(orth=False, impulse='a', response='b')
    plt.suptitle('a对b的脉冲响应')
    irf.plot(orth=False, impulse='b', response='a')
    plt.suptitle('b对a的脉冲响应')
    plt.show()

