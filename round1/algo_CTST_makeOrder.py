from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import json
import statistics
import math
from algo_CTST_MR import Product

class Trader:
  def __init__(self):
    self.price_history = {}  # 存储每个产品的价格历史
    # 不需要自己维护position了，因为TradingState中已经有了
    self.MAX_POS = 50
    self.WINDOW_SIZE = 20  # 滑动窗口大小
    self.BUY_THRESHOLD = 0.1  # 买入阈值：低于最高点的80%
    self.SELL_THRESHOLD = 0.4  # 卖出阈值：高于最低点的20%
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
    self.BUY_DIFFERENCE = 1   # 挂出买单与订单流最小值的价差
    self.SELL_DIFFERENCE = 1  # 挂出卖单与订单流最大值的价差
    self.LIMIT = {Product.RAINFOREST_RESIN: 50, Product.KELP: 50, Product.SQUID_INK: 50}  # 持仓限制

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


  # 做市
  def market_make(self,
    product: str,                           # 交易品种
    orders: List[Order],                    # 待填充的订单列表，函数向其中追加新订单
    bid: int,                               # 买单挂单价格
    ask: int,                               # 卖单挂单价格
    position: int,                          # 当前持仓量
    buy_order_volume: int,                  # 当前时间戳内已生成的买单量
    sell_order_volume: int)-> (int, int):   # 当前时间戳内已生成的卖单量
    # 计算可挂买单量（考虑已挂买单量）
    buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
    if buy_quantity > 0:
      orders.append(Order(product, bid, buy_quantity))  # 挂买单

    # 计算可挂卖单量（考虑已挂卖单量）
    sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
    if sell_quantity > 0:
      orders.append(Order(product, ask, -sell_quantity))  # 挂卖单

    return buy_order_volume + buy_quantity, sell_order_volume + sell_quantity

  # 有方向的做市逻辑(emm起名smd难搞）
  def make_tendency_orders (self, product : str, state: TradingState, tendency : float,
                            position: int, buy_order_volume: int, sell_order_volume: int) ->List[Order]:

    orders: List[Order] = []
    order_depth = state.order_depths[product]
    # 最高卖价
    best_bid = max(order_depth.buy_orders.keys(), default=0)
    # 最低买价
    best_ask = min(order_depth.sell_orders.keys(), default=0)

    # 最大持仓比例
    max_position_ratio = 0.5

    # 动态计算价差
    # 根据产品特性差异化参数
    if product == "SQUID_INK":
      spread_ratio = 0.003  # 高波动品种扩大价差
    else:
      spread_ratio = 0.001
    # spread_ratio = 0.003

    sell_diff = int(best_bid * spread_ratio)
    buy_diff = int(best_ask * spread_ratio)

    # 趋势自适应
    if tendency < 5:                    # 上涨趋势
      buy_diff = max(0, buy_diff - 1)   # 缩小买单价差
      sell_diff += 1                    # 扩大卖单价差
    elif tendency >= 5:
      buy_diff += 1
      sell_diff = max(0, sell_diff - 1)


    # 挂单价格
    sell_price = best_bid + sell_diff   # 高于别人挂的买单最高价挂卖单
    buy_price = best_ask - buy_diff


    # 持仓限制
    max_position = int(self.LIMIT[product] * max_position_ratio)

    # 计算有效挂单量
    valid_buy = max(0, min(
      max_position - position - buy_order_volume,  # 基于策略仓位限制
      self.LIMIT[product] - position - buy_order_volume  # 硬性仓位限制
    ))

    valid_sell = max(0, min(
      max_position + position - sell_order_volume,
      self.LIMIT[product] + position - sell_order_volume
    ))

    # 生成订单
    if valid_buy > 0 :
      orders.append(Order(product, buy_price, valid_buy))
    if valid_sell > 0 :
      orders.append(Order(product, sell_price, -valid_sell))

    return orders

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
      make_orders: List[Order] = []

      # 获取当前持仓，使用state.position
      current_pos = state.position.get(product, 0)
      # print(f"{product}  current_pos: {current_pos:.2f}")

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
          ask_volume = abs(order_depth.sell_orders[best_ask])  # 获取最优卖价档位的可用数量
          bid_volume = abs(order_depth.buy_orders[best_bid])
          # 首先确保市场上有买单和卖单（市场有流动性）
          if best_bid and best_ask:
            # 买入条件：价格低于固定最高价的80%且在上升
            if (mid_price < max_price * self.BUY_THRESHOLD and
              price_direction <5  and
              current_pos < self.MAX_POS):
              # ask_volume = abs(order_depth.sell_orders[best_ask]) # 获取最优卖价档位的可用数量
              print(f"{product}  ask_volume: {ask_volume:.2f}")
              buy_volume = min(ask_volume, self.MAX_POS - current_pos) # 根据当前持仓和最大持仓限制买入数量
              # 如果买入数量大于0，则创建买入订单
              if buy_volume > 0:
                orders.append(Order(product, best_ask, buy_volume)) # 创建买入订单
                print(f"{product} BUY {buy_volume} @ {best_ask} Direction: {price_direction:.2f}")

            # 卖出条件：价格高于固定最低价的120%且在下降
            elif (mid_price > min_price * self.SELL_THRESHOLD and
                price_direction >= 5 and
                current_pos > -self.MAX_POS):
              # bid_volume = abs(order_depth.buy_orders[best_bid])
              print(f"{product}  bid_volume: {bid_volume:.2f}")
              sell_volume = min(bid_volume, self.MAX_POS + current_pos)
              if sell_volume > 0:
                orders.append(Order(product, best_bid, -sell_volume))
                print(f"{product} SELL {sell_volume} @ {best_bid} Direction: {price_direction:.2f}")

          # 做市
          make_orders= self.make_tendency_orders(product, state, price_direction,
                                                  current_pos, bid_volume,  ask_volume)
        # print("make_orders", make_orders)
    result[product] = orders  + make_orders

    traderData = json.dumps({
      "price_history": self.price_history
    })

    return result, 0, traderData