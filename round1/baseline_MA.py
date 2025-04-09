from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import json
import statistics

class Trader:
  def __init__(self):
    self.price_history = {}  # 存储每个产品的价格历史
    # 不需要自己维护position了，因为TradingState中已经有了
    self.MA_SHORT = 5
    self.MA_LONG = 10
    self.MAX_POS = 20

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