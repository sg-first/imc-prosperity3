from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List
import string
import jsonpickle
import numpy as np
import math


class Product:
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"


PARAMS = {
    Product.MAGNIFICENT_MACARONS:{
        "make_edge": 1.8,          # 初始报价边缘
        "make_min_edge": 1.3,       # 最小允许边缘
        "init_make_edge": 2,      # 初始边缘
        "volume_avg_timestamp":5,# 成交量计算窗口
        "volume_bar":20,          # 成交量调整阈值
        "dec_edge_discount":0.8,  # 减价折扣系数
        "step_size":0.2           # 边缘调整步长
    }
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {Product.MAGNIFICENT_MACARONS: 75}
        self.sunlight_history = []  # 新增阳光指数历史数据存储
        self.price_history = []  # 价格历史数据
        self.PRICE_WINDOW = 30



    def MAGNIFICENT_MACARONS_implied_bid_ask(
            self,
            observation: ConversionObservation,
    ) -> (float, float):
        # 买价减去出口税
        return observation.bidPrice - observation.exportTariff - observation.transportFees - 1, observation.askPrice + observation.importTariff + observation.transportFees

    def MAGNIFICENT_MACARONS_adap_edge(
            self,
            timestamp: int,
            curr_edge: float,
            position: int,
            traderObject: dict
    ) -> float:
        if timestamp == 0:
            traderObject["MAGNIFICENT_MACARONS"] = {
                "curr_edge": self.params[Product.MAGNIFICENT_MACARONS]["init_make_edge"],
                "volume_history": [],
                "optimized": False
            }
            return self.params[Product.MAGNIFICENT_MACARONS]["init_make_edge"]


        # Timestamp not 0
        if "MAGNIFICENT_MACARONS" not in traderObject:
            traderObject["MAGNIFICENT_MACARONS"] = {
                "curr_edge": curr_edge,
                "volume_history": [],
                "optimized": False
            }
        traderObject["MAGNIFICENT_MACARONS"]["volume_history"].append(abs(position))
        if len(traderObject["MAGNIFICENT_MACARONS"]["volume_history"]) > self.params[Product.MAGNIFICENT_MACARONS][
            "volume_avg_timestamp"]:
            traderObject["MAGNIFICENT_MACARONS"]["volume_history"].pop(0)

        if len(traderObject["MAGNIFICENT_MACARONS"]["volume_history"]) < self.params[Product.MAGNIFICENT_MACARONS][
            "volume_avg_timestamp"]:
            return curr_edge
        elif not traderObject["MAGNIFICENT_MACARONS"]["optimized"]:
            volume_avg = np.mean(traderObject["MAGNIFICENT_MACARONS"]["volume_history"])

            # Bump up edge if consistently getting lifted full size

            if volume_avg >= self.params[Product.MAGNIFICENT_MACARONS]["volume_bar"]:
                traderObject["MAGNIFICENT_MACARONS"]["volume_history"] = []
                traderObject["MAGNIFICENT_MACARONS"]["curr_edge"] = curr_edge + \
                                                                    self.params[Product.MAGNIFICENT_MACARONS][
                                                                        "step_size"]
                return curr_edge + self.params[Product.MAGNIFICENT_MACARONS]["step_size"]



            # Decrement edge if more cash with less edge, included discount

            elif self.params[Product.MAGNIFICENT_MACARONS]["dec_edge_discount"] * \
                    self.params[Product.MAGNIFICENT_MACARONS]["volume_bar"] * (
                    curr_edge - self.params[Product.MAGNIFICENT_MACARONS]["step_size"]) > volume_avg * curr_edge:
                if curr_edge - self.params[Product.MAGNIFICENT_MACARONS]["step_size"] > \
                        self.params[Product.MAGNIFICENT_MACARONS]["make_min_edge"]:
                    traderObject["MAGNIFICENT_MACARONS"]["volume_history"] = []
                    traderObject["MAGNIFICENT_MACARONS"]["curr_edge"] = curr_edge - \
                                                                        self.params[Product.MAGNIFICENT_MACARONS][
                                                                            "step_size"]
                    traderObject["MAGNIFICENT_MACARONS"]["optimized"] = True
                    return curr_edge - self.params[Product.MAGNIFICENT_MACARONS]["step_size"]
                else:
                    traderObject["MAGNIFICENT_MACARONS"]["curr_edge"] = self.params[Product.MAGNIFICENT_MACARONS][
                        "make_min_edge"]
                    return self.params[Product.MAGNIFICENT_MACARONS]["make_min_edge"]

        traderObject["MAGNIFICENT_MACARONS"]["curr_edge"] = curr_edge
        return curr_edge

    def get_weighted_mid(self, order_depth):
        """计算中间价"""
        buy_values = [price * abs(qty) for price, qty in order_depth.buy_orders.items()]
        sell_values = [price * abs(qty) for price, qty in order_depth.sell_orders.items()]
        total_qty = sum(abs(qty) for qty in order_depth.buy_orders.values()) + \
                    sum(abs(qty) for qty in order_depth.sell_orders.values())

        if total_qty > 0:
            return (sum(buy_values) + sum(sell_values)) / total_qty
        return (max(order_depth.buy_orders.keys(), default=0) +
                min(order_depth.sell_orders.keys(), default=0)) / 2

    def update_price_stats(self, mid_price):
        """根据窗口计算马卡龙的均值和标准差，rolling.std()不懂的都给我去死"""
        self.price_history.append(mid_price)
        if len(self.price_history) > self.PRICE_WINDOW:
            self.price_history.pop(0)

        if len(self.price_history) <= self.PRICE_WINDOW:
            self.mean_makalong = np.mean(self.price_history)
            self.sigma = np.std(self.price_history)



    def MAGNIFICENT_MACARONS_arb_take(self, order_depth, observation, position):
        """主动吃单策略"""
        orders = []

        buy_order_volume = 0
        sell_order_volume = 0


        CSI = 55

        # 计算市场指标
        weighted_mid = self.get_weighted_mid(order_depth)
        self.update_price_stats(weighted_mid)

        # 策略逻辑
        position_limit = self.LIMIT[Product.MAGNIFICENT_MACARONS]
        max_buy = position_limit - position
        max_sell = position_limit + position

        # 阳光指数策略
        if observation.sunlightIndex < CSI:  # 看涨信号
            # 吃卖单逻辑
            for price in sorted(order_depth.sell_orders.keys()):
                if price < (self.mean_makalong - 1 * self.sigma) and buy_order_volume < max_buy:
                    qty = min(abs(order_depth.sell_orders[price]), max_buy - buy_order_volume)
                    orders.append(Order(Product.MAGNIFICENT_MACARONS, price, qty))
                    buy_order_volume += qty

            # 止盈逻辑
            for price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if price > (self.mean_makalong + 1.8 * self.sigma) and sell_order_volume < max_sell:
                    qty = min(abs(order_depth.buy_orders[price]), max_sell - sell_order_volume)
                    orders.append(Order(Product.MAGNIFICENT_MACARONS, price, -qty))
                    sell_order_volume += qty


        return orders, buy_order_volume, sell_order_volume

    def MAGNIFICENT_MACARONS_arb_clear(
            self,
            position: int
    ) -> int:
        conversions = -position

        conversions = max(-10, min(10, conversions))
        return conversions


# 主要是做买卖单，提供流动性。
    def MAGNIFICENT_MACARONS_arb_make(
            self,
            order_depth: OrderDepth,
            observation: ConversionObservation,
            position: int,
            edge: float,
            buy_order_volume: int,
            sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        position_limit = self.LIMIT[Product.MAGNIFICENT_MACARONS]

        implied_bid, implied_ask = self.MAGNIFICENT_MACARONS_implied_bid_ask(observation)

        bid = implied_bid - edge
        ask = implied_ask + edge
        # Round1:第一次比价！！！
        # 算出隐含中间价格，再用这个价格算出个有竞争力的卖价。
        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        aggressive_ask = foreign_mid - 1
        # 如果这个卖价比隐含卖价+edge还多，哎呀，就是你了，我的老baby
        if aggressive_ask >= implied_ask + self.params[Product.MAGNIFICENT_MACARONS]['make_min_edge']:
            ask = aggressive_ask
            print("AGGRESSIVE")
            print(f"ALGO ASK: {round(ask)}")
            print(f"ALGO BID: {round(bid)}")
        else:
            print(f"ALGO ASK: {round(ask)}")
            print(f"ALGO BID: {round(bid)}")

        # Round2:第二次比价！！！参考大单的价格
        filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 25]
        filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 15]

        # 如果有大单，且大单减1都比隐含卖价高，哎哟就是你了，你就是我的新宝贝卖价。就是要跟地主抢生意
        if len(filtered_ask) > 0 and ask > filtered_ask[0]:
            if filtered_ask[0] - 1 > implied_ask:
                ask = filtered_ask[0] - 1
            else:
                ask = implied_ask + edge



        # 跟地主比买价！我比地主最低价+1都比我的心里预期价格要低的话，哎哟，你就是最优的买价。
        if len(filtered_bid) > 0 and bid < filtered_bid[0]:
            if filtered_bid[0] + 1 < implied_bid:
                bid = filtered_bid[0] + 1
            else:
                bid = implied_bid - edge

        print(f"IMPLIED_BID: {implied_bid}")
        print(f"IMPLIED_ASK: {implied_ask}")
        print(f"FOREIGN ASK: {observation.askPrice}")
        print(f"FOREIGN BID: {observation.bidPrice}")

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(Product.MAGNIFICENT_MACARONS, round(bid), buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(Product.MAGNIFICENT_MACARONS, round(ask), -sell_quantity))

        return orders, buy_order_volume, sell_order_volume






    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0

        if Product.MAGNIFICENT_MACARONS in self.params and Product.MAGNIFICENT_MACARONS in state.order_depths:
            if "MAGNIFICENT_MACARONS" not in traderObject:
                traderObject["MAGNIFICENT_MACARONS"] = {
                    "curr_edge": self.params[Product.MAGNIFICENT_MACARONS]["init_make_edge"],
                    "volume_history": [],
                    "optimized": False
                }
                # 先获取仓位
            MAGNIFICENT_MACARONS_position = (
                state.position[Product.MAGNIFICENT_MACARONS]
                if Product.MAGNIFICENT_MACARONS in state.position
                else 0
            )
            print(f"MAGNIFICENT_MACARONS POSITION: {MAGNIFICENT_MACARONS_position}")

            adap_edge = self.MAGNIFICENT_MACARONS_adap_edge(
                state.timestamp,
                traderObject["MAGNIFICENT_MACARONS"]["curr_edge"],
                MAGNIFICENT_MACARONS_position,
                traderObject,
            )

            MAGNIFICENT_MACARONS_take_orders, buy_order_volume, sell_order_volume = self.MAGNIFICENT_MACARONS_arb_take(
                state.order_depths[Product.MAGNIFICENT_MACARONS],
                state.observations.conversionObservations[Product.MAGNIFICENT_MACARONS],
                MAGNIFICENT_MACARONS_position,
            )

            conversions = self.MAGNIFICENT_MACARONS_arb_clear(
                MAGNIFICENT_MACARONS_position
            )

            MAGNIFICENT_MACARONS_position += conversions


            MAGNIFICENT_MACARONS_make_orders, _, _ = self.MAGNIFICENT_MACARONS_arb_make(
                state.order_depths[Product.MAGNIFICENT_MACARONS],
                state.observations.conversionObservations[Product.MAGNIFICENT_MACARONS],
                MAGNIFICENT_MACARONS_position,
                adap_edge,
                buy_order_volume,
                sell_order_volume
            )

            result[Product.MAGNIFICENT_MACARONS] = (
                    MAGNIFICENT_MACARONS_take_orders + MAGNIFICENT_MACARONS_make_orders
            )

        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData

