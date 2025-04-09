from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import json
import statistics

class Trader:
  def __init__(self):
    self.price_history = {}  # 存储每个产品的价格历史
    # 不需要自己维护position了，因为TradingState中已经有了
    self.MAX_POS = 20
    self.WINDOW_SIZE = 20  # 滑动窗口大小
    self.BUY_THRESHOLD = 0.8  # 买入阈值：低于最高点的80%
    self.SELL_THRESHOLD = 0.2  # 卖出阈值：高于最低点的20%
    self.PRICE_LIMITS = { # 为不同产品设置固定的极值点参数
      "RAINFOREST_RESIN": {
        "max": 10003.5,  # 最高价
        "min": 9996.5,   # 最低价
      },
      "SQUID_INK": {
        "max": 2183.0,   # 最高价
        "min": 1956.5,   # 最低价
      },
      "KELP":{
        "max": 2035.5,  # 最高价
        "min": 2013.5,   # 最低价
      }
    }

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
  
  def get_price_direction(self, prices: list) -> float:
    """计算价格方向
    使用窗口内第一个点和最后一个点计算整体趋势斜率
    """
    if len(prices) < self.WINDOW_SIZE:  # 数据不足一个完整窗口
      if len(prices) < 2:  # 至少需要两个点
        return 0
      else: # 使用可用数据的首尾两点
        return prices[-1] - prices[0]
    
    # 使用窗口内的首尾两点计算趋势
    window = prices[-self.WINDOW_SIZE:]  # 取最近的WINDOW_SIZE个价格
    return window[-1] - window[0]  # 终点减起点得到趋势

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
      
      # 控制滑动窗口大小
      if len(self.price_history[product]) > self.WINDOW_SIZE:
        self.price_history[product].pop(0)
      
      # 当有足够的历史数据时执行策略
      if len(self.price_history[product]) >= 2:
        price_direction = self.get_price_direction(self.price_history[product])
        
        # 使用固定的极值点参数
        price_limits = self.PRICE_LIMITS.get(product)
        if price_limits:
          max_price = price_limits["max"]
          min_price = price_limits["min"]
          
          best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
          best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0
          
          # 首先确保市场上有买单和卖单（市场有流动性）
          if best_bid and best_ask:
            # 买入条件：价格低于固定最高价的80%且在上升
            if (mid_price < max_price * self.BUY_THRESHOLD and 
              price_direction > 0 and 
              current_pos < self.MAX_POS):
              ask_volume = abs(order_depth.sell_orders[best_ask]) # 获取最优卖价档位的可用数量
              buy_volume = min(ask_volume, self.MAX_POS - current_pos) # 根据当前持仓和最大持仓限制买入数量
              # 如果买入数量大于0，则创建买入订单
              if buy_volume > 0:
                orders.append(Order(product, best_ask, buy_volume)) # 创建买入订单
                print(f"{product} BUY {buy_volume} @ {best_ask} Direction: {price_direction:.2f}")
            
            # 卖出条件：价格高于固定最低价的120%且在下降
            elif (mid_price > min_price * self.SELL_THRESHOLD and 
                price_direction < 0 and 
                current_pos > -self.MAX_POS):
              bid_volume = abs(order_depth.buy_orders[best_bid])
              sell_volume = min(bid_volume, self.MAX_POS + current_pos)
              if sell_volume > 0:
                orders.append(Order(product, best_bid, -sell_volume))
                print(f"{product} SELL {sell_volume} @ {best_bid} Direction: {price_direction:.2f}")
        
    result[product] = orders
    
    traderData = json.dumps({
      "price_history": self.price_history
    })
    
    return result, 0, traderData