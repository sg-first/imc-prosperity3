import pandas as pd
import matplotlib.pyplot as plt

day = -1
market_data = pd.read_csv(f"D:/Program Files/IMC/round-2-island-data-bottle/prices_round_2_day_{day}.csv", sep=";", header=0)

# 筛选各个产品的数据
djembes = market_data[market_data['product'] == 'DJEMBES']
croissants = market_data[market_data['product'] == 'CROISSANTS']
jams = market_data[market_data['product'] == 'JAMS']
picnic_basket1 = market_data[market_data['product'] == 'PICNIC_BASKET1']
picnic_basket2 = market_data[market_data['product'] == 'PICNIC_BASKET2']

# 合并数据，确保每个时间步都有对应的价格
merged_data = pd.merge(picnic_basket1, croissants, on='timestamp', suffixes=('_basket1', '_croissants'))
merged_data = pd.merge(merged_data, jams, on='timestamp', suffixes=('', '_jams'))
merged_data = pd.merge(merged_data, djembes, on='timestamp', suffixes=('', '_djembes'))
merged_data = pd.merge(merged_data, picnic_basket2, on='timestamp', suffixes=('', '_basket2'))

# 重命名列以避免冲突
merged_data.rename(columns={
    'mid_price_basket1': 'mid_price_picnic_basket1',
    'mid_price_croissants': 'mid_price_croissants',
    'mid_price': 'mid_price_jams',
    'mid_price_djembes': 'mid_price_djembes',
    'mid_price_basket2': 'mid_price_picnic_basket2'
}, inplace=True)

# 计算每个时间步的价差
merged_data['spread1'] = merged_data['mid_price_picnic_basket1'] - (merged_data['mid_price_croissants'] * 6 + merged_data['mid_price_jams'] * 3 + merged_data['mid_price_djembes'])
merged_data['spread2'] = merged_data['mid_price_picnic_basket2'] - (merged_data['mid_price_croissants'] * 4 + merged_data['mid_price_jams'] * 2)

# 打印价差
print("每个时间步的价差:")
print(merged_data[['timestamp', 'spread1', 'spread2']].head(50))  # 打印前50行

# 使用pivot_table确保每个标的都读出一组中间价
pivot_table = market_data.pivot_table(index='timestamp', columns='product', values='mid_price', aggfunc='first')

# 计算每个时间步的价差
pivot_table['spread1'] = pivot_table['PICNIC_BASKET1'] - (pivot_table['CROISSANTS'] * 6 + pivot_table['JAMS'] * 3 + pivot_table['DJEMBES'])
pivot_table['spread2'] = pivot_table['PICNIC_BASKET2'] - (pivot_table['CROISSANTS'] * 4 + pivot_table['JAMS'] * 2)

# 绘制图表
plt.figure(figsize=(12, 6), dpi=100)  # 设置图表尺寸和分辨率

# 绘制spread1，设置图线宽度
plt.plot(pivot_table.index, pivot_table['spread1'], label='PICNIC_BASKET1 Spread', linewidth=0.5)

# 绘制spread2，设置图线宽度
plt.plot(pivot_table.index, pivot_table['spread2'], label='PICNIC_BASKET2 Spread', linewidth=0.5)

# 设置图表标题和坐标轴标签
plt.title('Picnic Basket Spreads Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Spread')
plt.legend()

# 旋转x轴标签以提高可读性
plt.xticks(rotation=45)

# 显示网格线
plt.grid(True)

# 显示图表
plt.tight_layout()
plt.show()