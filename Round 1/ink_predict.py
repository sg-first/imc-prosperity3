import numpy as np

from datamodel import OrderDepth, UserId, TradingState, Order,Trade
from typing import List
import json
import statistics

class Product:
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    RAINFOREST_RESIN = 'RAINFOREST_RESIN'


# PARAMS = {
#     Product.AMETHYSTS: {
#         "fair_value": 10000,
#         "take_width": 1,
#         "clear_width": 0,
#         # for making
#         "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
#         "join_edge": 2,  # joins orders within this edge
#         "default_edge": 4,
#         "soft_position_limit": 10,
#     },
#     Product.STARFRUIT: {
#         "take_width": 1,
#         "clear_width": 0,
#         "prevent_adverse": True, # 是否防止逆向交易，true为防止与大额订单成交，避免因一次性吃单过多导致不利价格波动。
#         "adverse_volume": 15,
#         "reversion_beta": -0.229,
#         "disregard_edge": 1,
#         "join_edge": 0,
#         "default_edge": 1,
#     },
# }

class Trader:
    def __init__(self):
        self.price_history = {}  # 存储每个产品的价格历史
        # 不需要自己维护position了，因为TradingState中已经有了
        self.MA_SHORT = 5
        self.MA_LONG = 10
        self.MAX_POS = 20

        # 通过VAR模型对回测数据DAY-2进行分析得出的ink差分价格1预测方程的滞后n阶的kelp对应的系数，L1,L2……
        self.kelp_coeffs = np.array([-0.56672919,-0.39499563,-0.31645824,-0.18454702,-0.13158352,-0.15589962,-0.11816887])
        self.kelp_price = []
        self.kelp_diff_history = []
        self.LIMIT = {Product.KELP:50, Product.SQUID_INK:50, Product.RAINFOREST_RESIN:50}

    def get_current_kelp_price(self, trade:Trade, order_depth: OrderDepth,market_trades) -> float:
        """获取当前KELP价格（优先使用最新交易价，其次使用订单簿中位数）"""
        # 从市场交易记录获取
        kelp_trades = market_trades.get("KELP", trade.price)
        if kelp_trades:
            latest_trade = max(kelp_trades, key=lambda t: t.timestamp)
            return latest_trade.price

        # 从订单簿计算中位数
        if len(order_depth.buy_orders) != 0 and len(order_depth.sell_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2

        # 无数据可用
        return None


    def get_predictions_ink_tendency(self,current_kelp_price):
        current_kelp_price
        self.kelp_price.append(current_kelp_price)

        # 计算当前差分
        if len(self.kelp_price) < 2:
            return 0.0
        else:
            current_diff = self.kelp_price[-1] - self.kelp_price[-2]

        # 维护固定长度历史窗口
        if len(self.kelp_diff_history) >= 7:
            self.kelp_diff_history.pop(0)
        self.kelp_diff_history.append(current_diff)

        # 当数据不足时返回0
        if len(self.kelp_diff_history) < 7:
            return 0.0

        # 计算预测值
        window = np.array(self.kelp_diff_history[-7:])[::-1]  # 倒序确保L1在前
        return np.dot(self.kelp_coeffs, window)



    def get_mid_price(self, order_depth: OrderDepth) -> float:
        """计算中间价"""
        # 获取最优买价（买方愿意买入的最高价格）
        best_bid = max(order_depth.buy_orders.keys()) if len(order_depth.buy_orders) > 0 else 0
        # 获取最优卖价（卖方愿意卖出的最低价格）
        best_ask = min(order_depth.sell_orders.keys()) if len(order_depth.sell_orders) > 0 else 0
        # 如果买价或卖价有一个为0，直接返回非0的那个价格
        if best_bid == 0 or best_ask == 0:
            return best_bid + best_ask
        # 返回买卖价的中间值
        return (best_bid + best_ask) / 2

    def run(self, state: TradingState):
        result = {}
        # 遍历所有产品
        for product in state.order_depths:
            # 如果产品不在价格历史中，初始化价格历史
            if product not in self.price_history:
                self.price_history[product] = []
            # 获取当前产品的订单深度
            order_depth = state.order_depths[product]
            orders: List[Order] = []

            # 获取当前持仓，使用state.position
            current_pos = state.position.get(product, 0)

            # 计算当前中间价并更新价格历史
            mid_price = self.get_mid_price(order_depth)
            self.price_history[product].append(mid_price)

            if len(self.price_history[product]) > self.MA_LONG:
                self.price_history[product].pop(0)

            if len(self.price_history[product]) >= self.MA_LONG:
                ma_short = statistics.mean(self.price_history[product][-self.MA_SHORT:])
                ma_long = statistics.mean(self.price_history[product][-self.MA_LONG:])

                best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
                best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0

                if best_bid and best_ask:
                    # 使用state中的position
                    if ma_short > ma_long and current_pos < self.MAX_POS:
                        ask_volume = abs(order_depth.sell_orders[best_ask])
                        buy_volume = min(ask_volume, self.MAX_POS - current_pos)
                        if buy_volume > 0:
                            orders.append(Order(product, best_ask, buy_volume))
                            # 不需要更新self.position了

                    elif ma_short < ma_long and current_pos > -self.MAX_POS:
                        bid_volume = abs(order_depth.buy_orders[best_bid])
                        sell_volume = min(bid_volume, self.MAX_POS + current_pos)
                        if sell_volume > 0:
                            orders.append(Order(product, best_bid, -sell_volume))
                            # 不需要更新self.position了

            # 可以利用state中的其他信息来优化策略
            # 1. 使用market_trades查看市场成交情况
            if product in state.market_trades:
                recent_trades = state.market_trades[product]
                # 可以分析最近的成交来调整策略

            # 2. 使用own_trades查看自己的成交情况
            if product in state.own_trades:
                my_trades = state.own_trades[product]
                # 可以分析自己的成交来优化策略

            result[product] = orders

        # 可以使用traderData来保存一些状态信息
        traderData = json.dumps({
            "price_history": self.price_history
        })

        return result, 0, traderData