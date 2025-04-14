# mean_reversion.py
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from matplotlib import rcParams


class MeanReversionAnalyzer:
    """
    均值回归分析工具类

    功能包含：
    - ADF平稳性检验
    - Hurst指数计算
    - 半衰期计算
    - 可视化分析

    使用方法：
    >>> from MeanReversion import MeanReversionAnalyzer
    >>> analyzer = MeanReversionAnalyzer(spread_series)
    >>> analyzer.check_stationarity()
    >>> analyzer.plot_analysis()
    """

    def __init__(self, spread_series):
        """
        初始化方法
        :param spread_series: pd.Series 价差时间序列
        """
        self.spread = spread_series.dropna().copy()
        self._setup_plot_style()

    def _setup_plot_style(self):
        """配置绘图样式"""
        rcParams['font.sans-serif'] = ['SimHei']
        rcParams['axes.unicode_minus'] = False
        plt.style.use('ggplot')

    def check_stationarity(self):
        """执行ADF平稳性检验"""
        result = adfuller(self.spread)
        print('ADF统计量:', result[0])
        print('p-value:', result[1])
        print("序列平稳" if result[1] < 0.05 else "序列非平稳")

    def hurst_exponent(self):
        """修正后的Hurst指数计算"""
        spread = self.spread.dropna().values
        if len(spread) < 100:
            return np.nan

        # 调整滞后范围（从1开始）
        max_lag = min(100, len(spread) // 2)
        lags = range(1, max_lag + 1)

        # 预计算对数范围
        log_lags = []
        log_tau = []

        for lag in lags:
            if 2 * lag > len(spread):
                continue

            # 正确计算滞后差值
            diffs = spread[lag:] - spread[:-lag]

            # 过滤零值（避免log(0)）
            std = np.std(diffs)
            if std < 1e-12:
                continue

            log_lags.append(np.log(lag))
            log_tau.append(np.log(std))

        if len(log_lags) < 2:
            return np.nan

        # 线性回归拟合
        slope, _ = np.polyfit(log_lags, log_tau, 1)
        return slope * 2
    
    def calculate_half_life(self):
        """计算均值回归半衰期"""
        if len(self.spread) < 3:
            return np.nan

        spread_lag = self.spread.shift(1).dropna()
        delta_spread = self.spread.diff().dropna()

        model = sm.OLS(delta_spread, sm.add_constant(spread_lag)).fit()
        if model.params[1] >= 0:
            return np.inf
        return -np.log(2) / model.params[1]

    def plot_analysis(self):
        """综合可视化分析"""
        plt.figure(figsize=(12, 6))

        # 计算指标
        H = self.hurst_exponent()
        half_life = self.calculate_half_life()
        mean_val = self.spread.mean()
        std_val = self.spread.std()

        # 绘制价差序列
        plt.plot(self.spread.index, self.spread.values,
                 label=f'Hurst指数: {H:.2f}',
                 color='steelblue',
                 alpha=0.7)

        # 绘制统计线
        plt.axhline(mean_val, color='crimson', ls='--',
                    label=f'均值 ({mean_val:.2f})')
        plt.fill_between(self.spread.index,
                         mean_val - std_val,
                         mean_val + std_val,
                         color='grey', alpha=0.2,
                         label='±1标准差')

        # 信息标注
        plt.text(0.05, 0.95,
                 f"半衰期: {half_life:.1f}周期\n波动率: {std_val:.2f}",
                 transform=plt.gca().transAxes,
                 va='top',
                 bbox=dict(facecolor='white', alpha=0.8))

        # 图表美化
        plt.title("价差均值回归分析", fontsize=14)
        plt.xlabel("时间", fontsize=12)
        plt.ylabel("价差值", fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, ls='--', alpha=0.5)
        plt.tight_layout()
        plt.show()


# 使用示例 ---------------------------------------------------
if __name__ == '__main__':
    # 示例数据
    np.random.seed(42)
    test_data = pd.Series(
        np.cumsum(np.random.randn(200)) + 10,
        index=pd.date_range('2023-01-01', periods=200)
    )

    # 使用类进行分析
    analyzer = MeanReversionAnalyzer(test_data)
    analyzer.check_stationarity()
    print(f"Hurst指数: {analyzer.hurst_exponent():.2f}")
    print(f"半衰期: {analyzer.calculate_half_life():.1f}")
    analyzer.plot_analysis()