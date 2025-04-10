from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import json
import jsonpickle
import numpy as np
import statistics

class Trader:
  def __init__(self):
    self.price_history = {}  # 存储每个产品的价格历史
    # 不需要自己维护position了，因为TradingState中已经有了
    self.MAX_POS = 20
    self.WINDOW_SIZE = 20  # 滑动窗口大小
    self.WINDOW_SIZE2 = 200
    self.KELP_MEAN = None
    self.BUY_THRESHOLD = 0.4  # 买入阈值：低于最低价+极差的20%
    self.SELL_THRESHOLD = 0.8  # 卖出阈值：高于最低价+极差的80%
    self.PRICE_LIMITS = { # 为不同产品设置固定的极值点参数
      "RAINFOREST_RESIN": {
        "max": 10003.5,  # 最高价
        "min": 9996.5,   # 最低价
      },
      "SQUID_INK": {
        "max": 2181.0,   # 最高价
        "min": 1814,   # 最低价
      },
        # 我再他妈的确认以下，第负二天1964去到2175，很多时间是1965附近。然后第负一天慢慢跌到1950附近很多时间都是，涨到2038，立刻跌到1930，然后回到1950附近，然后一直跌到1814
        # kelp大部分都在2020-2030附近
      "KELP":{
        "max": 2035.5,  # 最高价
        "min": 2013,   # 最低价
      }
    }
# 已经有中间价格了！
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

  def get_kelp_mean_price(self,product:str):
      if product in self.price_history and len(self.price_history[product]) > 0:
          if len(self.price_history[product]) > self.WINDOW_SIZE2:
              self.price_history[product].pop(0)  # 移除最旧的数据
          self.KELP_MEAN = np.mean(self.price_history[product])

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
          current_pos = state.position.get(product, 0)

          # 计算当前中间价并更新价格历史
          mid_price = self.get_mid_price(order_depth)
          self.price_history[product].append(mid_price)

          if product == "RAINFOREST_RESIN":  # 使用均值回归策略
              mr_trader = MRTrader(PARAMS)
              orders = handle_resin_trading(mr_trader, state)

          elif product == "KELP":
              # KELP专用逻辑
              self.get_kelp_mean_price(product)
              if self.KELP_MEAN is not None and len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
                  best_ask = min(order_depth.sell_orders.keys())
                  best_bid = max(order_depth.buy_orders.keys())

                  if best_ask < self.KELP_MEAN and current_pos < self.MAX_POS:
                      ask_volume = abs(order_depth.sell_orders[best_ask])
                      buy_volume = min(ask_volume, self.MAX_POS - current_pos)
                      if buy_volume > 0:
                          orders.append(Order(product, best_ask, buy_volume))
                          print(f"KELP BUY {buy_volume} @ {best_ask} (Mean: {self.KELP_MEAN:.2f})")

                  if best_bid > self.KELP_MEAN and current_pos > -self.MAX_POS:
                      bid_volume = abs(order_depth.buy_orders[best_bid])
                      sell_volume = min(bid_volume, self.MAX_POS + current_pos)
                      if sell_volume > 0:
                          orders.append(Order(product, best_bid, -sell_volume))
                          print(f"KELP SELL {sell_volume} @ {best_bid} (Mean: {self.KELP_MEAN:.2f})")
          else:
              # 其他产品逻辑
              if len(self.price_history[product]) > self.WINDOW_SIZE:
                  self.price_history[product].pop(0)

              if len(self.price_history[product]) >= 2:
                  price_direction = self.get_price_direction(self.price_history[product])
                  price_limits = self.PRICE_LIMITS.get(product)

                  if price_limits:
                      max_price = price_limits["max"]
                      min_price = price_limits["min"]
                      best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
                      best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0

                      if best_bid and best_ask:
                          if (mid_price < (max_price - min_price) * self.BUY_THRESHOLD + min_price and
                                  price_direction > 0 and
                                  current_pos < self.MAX_POS):
                              ask_volume = abs(order_depth.sell_orders[best_ask])
                              buy_volume = min(ask_volume, self.MAX_POS - current_pos)
                              if buy_volume > 0:
                                  orders.append(Order(product, best_ask, buy_volume))
                                  print(f"{product} BUY {buy_volume} @ {best_ask} Direction: {price_direction:.2f}")

                          elif (mid_price > (max_price - min_price) * self.SELL_THRESHOLD + min_price and
                                price_direction < 0 and
                                current_pos > -self.MAX_POS):
                              bid_volume = abs(order_depth.buy_orders[best_bid])
                              sell_volume = min(bid_volume, self.MAX_POS + current_pos)
                              if sell_volume > 0:
                                  orders.append(Order(product, best_bid, -sell_volume))
                                  print(f"{product} SELL {sell_volume} @ {best_bid} Direction: {price_direction:.2f}")

          result[product] = orders

      traderData = json.dumps({
          "price_history": self.price_history,
          "kelp_mean": self.KELP_MEAN
      })

      return result, 0, traderData

















def handle_resin_trading(trader, state: TradingState):
    # 从state获取当前仓位和订单簿
      resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
      order_depth = state.order_depths[Product.RAINFOREST_RESIN]  # 直接取订单簿
      
      resin_take_orders, buy_order_volume, sell_order_volume = (
          trader.take_orders(
              Product.RAINFOREST_RESIN,
              state.order_depths[Product.RAINFOREST_RESIN],
              trader.params[Product.RAINFOREST_RESIN]["fair_value"],
              trader.params[Product.RAINFOREST_RESIN]["take_width"],
              resin_position,
          )
      )

      # 平仓
      resin_clear_orders, buy_order_volume, sell_order_volume = (
          trader.clear_orders(
              Product.RAINFOREST_RESIN,
              state.order_depths[Product.RAINFOREST_RESIN],
              trader.params[Product.RAINFOREST_RESIN]["fair_value"],
              trader.params[Product.RAINFOREST_RESIN]["clear_width"],
              resin_position,
              buy_order_volume,
              sell_order_volume,
          )
      )

      # 做市
      resin_make_orders, _, _ = trader.make_orders(
          Product.RAINFOREST_RESIN,
          state.order_depths[Product.RAINFOREST_RESIN],
          trader.params[Product.RAINFOREST_RESIN]["fair_value"],
          resin_position,
          buy_order_volume,
          sell_order_volume,
          trader.params[Product.RAINFOREST_RESIN]["disregard_edge"],
          trader.params[Product.RAINFOREST_RESIN]["join_edge"],
          trader.params[Product.RAINFOREST_RESIN]["default_edge"],
          True,
          trader.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
      )

      # 合并订单并返回
      return resin_take_orders + resin_clear_orders + resin_make_orders

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,  # 设定商品的基准价格
        "take_width": 1,  # 允许吃单的价差阈值,当对手方订单价格与fair_value的偏离超过clear_width时吃单
        "clear_width": 0,  # 平仓订单的挂单价差范围,要平仓时在fair_value ± clear_width挂单。
        # for making
        "disregard_edge": 1,  # 忽略对手方订单的范围阈值，挂单时忽略对手方订单中与fair_value 过近的订单
        "join_edge": 2,  # 加入已有订单的价差容忍范围，当订单价格在fair_value ± join_edge内时挂相同价格的订单，否则挂更优价格。
        "default_edge": 4,  # 默认挂单价差,当没有可直接交易的订单时，以fair_value ± default_edge挂单。
        "soft_position_limit": 10,
    }
}


class MRTrader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {Product.RAINFOREST_RESIN: 50, Product.KELP: 50, Product.SQUID_INK : 50}

    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    # 平仓逻辑，在 fair_value ± clear_width 范围内挂单
    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    # 做市挂单逻辑，根据 disregard_edge/join_edge/default_edge 计算挂单价格
    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        # will penny all other levels with higher edge
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume


    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            result[Product.RAINFOREST_RESIN] = handle_resin_trading(self, state)

        conversions = 1
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData
