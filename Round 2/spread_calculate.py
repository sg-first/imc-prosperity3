import pandas as pd
import matplotlib.pyplot as plt

# 定义要读取的天数
days = [-1, 0, 1]

# 读取所有天数的数据并合并
all_market_data = pd.concat([
    pd.read_csv(f"D:/Program Files/IMC/Round 2/round-2-island-data-bottle/prices_round_2_day_{day}.csv", sep=";", header=0).assign(day=day)
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
merged_data['spread1'] = merged_data['mid_price_picnic_basket1'] - (merged_data['mid_price_croissants'] * 6 + merged_data['mid_price_jams'] * 3 + merged_data['mid_price_djembes'])

# PICNIC_BASKET2的价差：PICNIC_BASKET2价格 - (4*CROISSANTS + 2*JAMS)
merged_data['spread2'] = merged_data['mid_price_picnic_basket2'] - (merged_data['mid_price_croissants'] * 4 + merged_data['mid_price_jams'] * 2)

# 打印价差数据
print("每个时间步的价差:")
print(merged_data[['timestamp', 'day', 'spread1', 'spread2']].head(50))  # 打印前50行

# 创建一个新的列，将 timestamp 和 day 组合起来，确保每个时间戳和天数的组合都是唯一的
merged_data['timestamp_day'] = merged_data['timestamp'] + merged_data['day'] * 1000000

# 按新的组合列排序
merged_data.sort_values(by=['timestamp_day'], inplace=True)

# 绘制图表
plt.figure(figsize=(12, 6), dpi=100)  # 设置图表尺寸和分辨率

# 绘制spread1
plt.plot(merged_data['timestamp_day'], merged_data['spread1'], label='PICNIC_BASKET1 Spread', linewidth=0.8, color='blue')
plt.text(merged_data['timestamp_day'].iloc[-1], merged_data['spread1'].iloc[-1], 'Spread1', color='blue', ha='left')

# 绘制spread2
plt.plot(merged_data['timestamp_day'], merged_data['spread2'], label='PICNIC_BASKET2 Spread', linewidth=0.8, color='red', linestyle='--')
plt.text(merged_data['timestamp_day'].iloc[-1], merged_data['spread2'].iloc[-1], 'Spread2', color='red', ha='left')

# 设置图表标题和坐标轴标签
plt.title('Picnic Basket Spreads Over Time')
plt.xlabel('Timestamp + Day')
plt.ylabel('Spread')
plt.legend()

# 旋转x轴标签以提高可读性
plt.xticks(rotation=45)

# 显示网格线
plt.grid(True)

# 调整布局以避免标签重叠
plt.tight_layout()

# 保存图表
plt.savefig('spread_chart.png')

# 显示图表
plt.show()