from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math
from math import log, sqrt, exp
from statistics import NormalDist




class Product:

    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"


PARAMS = {

    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "mean_volatility": 0.19,
        #"threshold": 0.00163,# ？？？？？？？？？？？？？？？？？
        "strike": 9500,
        "starting_time_to_expiry": 5 / 252,
        "std_window": 6,
        "zscore_threshold": 1,
    },
}
class BlackScholes:
    @staticmethod
    # strike即行权价格，spot是标的物的市场价格
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    # 看跌期权，用不上其实
    def black_scholes_put(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * sqrt(time_to_expiry)
        put_price = strike * NormalDist().cdf(-d2) - spot * NormalDist().cdf(-d1)
        return put_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def gamma(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) / (spot * volatility * sqrt(time_to_expiry))

    @staticmethod
    def vega(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        # print(f"d1: {d1}")
        # print(f"vol: {volatility}")
        # print(f"spot: {spot}")
        # print(f"strike: {strike}")
        # print(f"time: {time_to_expiry}")
        return NormalDist().pdf(d1) * (spot * sqrt(time_to_expiry)) / 100

    @staticmethod
    # 二分法迭代求出隐含波动率
    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:# 理论价过高→需降低σ
                high_vol = volatility
            else:# 理论价过低→需提高σ
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility




class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {

            Product.VOLCANIC_ROCK: 400,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
        }

    # 已经有中间价格了！
    def get_mid_price(self, order_depth: OrderDepth) -> float:
        """计算中间价"""
        # 获取最优买价（买方愿意买入的最高价格）
        best_bid = (
            max(order_depth.buy_orders.keys()) if len(order_depth.buy_orders) > 0 else 0
        )
        # 获取最优卖价（卖方愿意卖出的最低价格）
        best_ask = (
            min(order_depth.sell_orders.keys())
            if len(order_depth.sell_orders) > 0
            else 0
        )
        # 如果买价或卖价有一个为0，直接返回非0的那个价格
        if best_bid == 0 or best_ask == 0:
            return best_bid + best_ask
        # 返回买卖价的中间值
        return (best_bid + best_ask) / 2


# 在生成COCONUT_COUPON订单的同时调整COCONUT的仓位，以确保整体的对冲策略生效。
    # 确保无论是COCONUT_COUPON的仓位变化，还是新增的COCONUT_COUPON订单，都能够及时通过调整COCONUT的仓位来对冲。
    def coconut_hedge_orders(
        self,
        coconut_order_depth: OrderDepth,
        coconut_coupon_order_depth: OrderDepth,
        coconut_coupon_orders: List[Order],
        coconut_position: int,
        coconut_coupon_position: int,
        delta: float,
    ) -> List[Order]:
        if coconut_coupon_orders == None or len(coconut_coupon_orders) == 0:
            coconut_coupon_position_after_trade = coconut_coupon_position
        else:# 算出期权的当前持仓
            coconut_coupon_position_after_trade = coconut_coupon_position + sum(
                order.quantity for order in coconut_coupon_orders
            )
        target_coconut_position = 0
# 计算需要对冲的标的物数量，即需要多少数量的标的物。如果此时position等于限制的position那就不用继续处理了
        if coconut_coupon_position_after_trade != 0 :
            target_coconut_position = -delta * coconut_coupon_position_after_trade

        if target_coconut_position == coconut_position:
            return []

#
        target_coconut_quantity = target_coconut_position - coconut_position

        orders: List[Order] = []
        if target_coconut_quantity > 0:
            # Buy COCONUT
            best_ask = min(coconut_order_depth.sell_orders.keys())
            quantity = min(
                abs(target_coconut_quantity),
                self.LIMIT[Product.VOLCANIC_ROCK] - coconut_position,
            )
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_ask, round(quantity)))

        elif target_coconut_quantity < 0:
            # Sell COCONUT
            best_bid = max(coconut_order_depth.buy_orders.keys())
            quantity = min(
                abs(target_coconut_quantity),
                self.LIMIT[Product.VOLCANIC_ROCK] + coconut_position,
            )
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_bid, -round(quantity)))

        return orders

# 该函数根据计算出的COCONUT_COUPON的波动率（volatility）和历史数据，决定是否买入或卖出COCONUT_COUPON。
    def coconut_coupon_orders(
        self,
        coconut_coupon_order_depth: OrderDepth,
        coconut_coupon_position: int,
        traderData: Dict[str, Any],
        volatility: float,
    ) -> List[Order]:# 控制历史隐含波动率的数据条数，有窗口限制。
        traderData["past_coupon_vol"].append(volatility)
        if (
            len(traderData["past_coupon_vol"])
            < self.params[Product.VOLCANIC_ROCK_VOUCHER_9500]["std_window"]
        ):
            return [],[]

        if (
            len(traderData["past_coupon_vol"])
            > self.params[Product.VOLCANIC_ROCK_VOUCHER_9500]["std_window"]
        ):
            traderData["past_coupon_vol"].pop(0)
# 万众瞩目的Z-score来了
        # 添加极小值防止除零
        std_dev = np.std(traderData["past_coupon_vol"]) or 1e-8
        vol_z_score = (volatility - self.params[Product.VOLCANIC_ROCK_VOUCHER_9500]["mean_volatility"]) / std_dev
        # print(f"vol_z_score: {vol_z_score}")
        # print(f"zscore_threshold: {self.params[Product.COCONUT_COUPON]['zscore_threshold']}")
        if vol_z_score >= self.params[Product.VOLCANIC_ROCK_VOUCHER_9500]["zscore_threshold"]:
            if coconut_coupon_position != -self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_9500]:
                target_coconut_coupon_position = -self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_9500]
                if len(coconut_coupon_order_depth.buy_orders) > 0:
                    best_bid = max(coconut_coupon_order_depth.buy_orders.keys())
                    target_quantity = abs(
                        target_coconut_coupon_position - coconut_coupon_position
                    )
                    quantity = min(
                        target_quantity,
                        abs(coconut_coupon_order_depth.buy_orders[best_bid]),
                    )
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(Product.VOLCANIC_ROCK_VOUCHER_9500, best_bid, -quantity)], []
                    else:
                        return [Order(Product.VOLCANIC_ROCK_VOUCHER_9500, best_bid, -quantity)], [
                            Order(Product.VOLCANIC_ROCK_VOUCHER_9500, best_bid, -quote_quantity)
                        ]

        elif vol_z_score <= -self.params[Product.VOLCANIC_ROCK_VOUCHER_9500]["zscore_threshold"]:
            if coconut_coupon_position != self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_9500]:
                target_coconut_coupon_position = self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_9500]
                if len(coconut_coupon_order_depth.sell_orders) > 0:
                    best_ask = min(coconut_coupon_order_depth.sell_orders.keys())
                    target_quantity = abs(
                        target_coconut_coupon_position - coconut_coupon_position
                    )
                    quantity = min(
                        target_quantity,
                        abs(coconut_coupon_order_depth.sell_orders[best_ask]),
                    )
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(Product.VOLCANIC_ROCK_VOUCHER_9500, best_ask, quantity)], []
                    else:
                        return [Order(Product.VOLCANIC_ROCK_VOUCHER_9500, best_ask, quantity)], [
                            Order(Product.VOLCANIC_ROCK_VOUCHER_9500, best_ask, quote_quantity)
                        ]

        return [], []

    def get_past_returns(
        self,
        traderObject: Dict[str, Any],
        order_depths: Dict[str, OrderDepth],
        timeframes: Dict[str, int],
    ):
        returns_dict = {}

        for symbol, timeframe in timeframes.items():
            traderObject_key = f"{symbol}_price_history"
            if traderObject_key not in traderObject:
                traderObject[traderObject_key] = []

            price_history = traderObject[traderObject_key]

            if symbol in order_depths:
                order_depth = order_depths[symbol]
                if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                    current_price = (
                        max(order_depth.buy_orders.keys())
                        + min(order_depth.sell_orders.keys())
                    ) / 2
                else:
                    if len(price_history) > 0:
                        current_price = float(price_history[-1])
                    else:
                        returns_dict[symbol] = None
                        continue
            else:
                if len(price_history) > 0:
                    current_price = float(price_history[-1])
                else:
                    returns_dict[symbol] = None
                    continue

            price_history.append(
                f"{current_price:.1f}"
            )  # Convert float to string with 1 decimal place

            if len(price_history) > timeframe:
                price_history.pop(0)

            if len(price_history) == timeframe:
                past_price = float(price_history[0])  # Convert string back to float
                returns = (current_price - past_price) / past_price
                returns_dict[symbol] = returns
            else:
                returns_dict[symbol] = None

        return returns_dict

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        past_returns_timeframes = {"GIFT_BASKET": 500}
        past_returns_dict = self.get_past_returns(
            traderObject, state.order_depths, past_returns_timeframes
        )

        result = {}
        conversions = 0

        if Product.VOLCANIC_ROCK_VOUCHER_9500 not in traderObject:
            traderObject[Product.VOLCANIC_ROCK_VOUCHER_9500] = {
                "prev_coupon_price": 0,
                "past_coupon_vol": [],
            }

        if (
            Product.VOLCANIC_ROCK_VOUCHER_9500 in self.params
            and Product.VOLCANIC_ROCK_VOUCHER_9500 in state.order_depths
        ):
            coconut_coupon_position = (
                state.position[Product.VOLCANIC_ROCK_VOUCHER_9500]
                if Product.VOLCANIC_ROCK_VOUCHER_9500 in state.position
                else 0
            )

            coconut_position = (
                state.position[Product.VOLCANIC_ROCK]
                if Product.VOLCANIC_ROCK in state.position
                else 0
            )
            # print(f"coconut_coupon_position: {coconut_coupon_position}")
            # print(f"coconut_position: {coconut_position}")
            coconut_order_depth = state.order_depths[Product.VOLCANIC_ROCK]
            coconut_coupon_order_depth = state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_9500]
            coconut_mid_price = (
                min(coconut_order_depth.buy_orders.keys())
                + max(coconut_order_depth.sell_orders.keys())
            ) / 2
            #
            coconut_coupon_mid_price = self.get_mid_price(
                coconut_coupon_order_depth
            )
            tte = (
                self.params[Product.VOLCANIC_ROCK_VOUCHER_9500]["starting_time_to_expiry"]
                - (state.timestamp) / 1000000 / 252
            )
            volatility = BlackScholes.implied_volatility(
                coconut_coupon_mid_price,
                coconut_mid_price,
                self.params[Product.VOLCANIC_ROCK_VOUCHER_9500]["strike"],
                tte,
            )
            delta = BlackScholes.delta(
                coconut_mid_price,
                self.params[Product.VOLCANIC_ROCK_VOUCHER_9500]["strike"],
                tte,
                volatility,
            )

            coconut_coupon_take_orders, coconut_coupon_make_orders = (
                self.coconut_coupon_orders(
                    state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_9500],
                    coconut_coupon_position,
                    traderObject[Product.VOLCANIC_ROCK_VOUCHER_9500],
                    volatility,
                )
            )

            coconut_orders = self.coconut_hedge_orders(
                state.order_depths[Product.VOLCANIC_ROCK],
                state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_9500],
                coconut_coupon_take_orders,
                coconut_position,
                coconut_coupon_position,
                delta,
            )

            if coconut_coupon_take_orders != None or coconut_coupon_make_orders != None:
                result[Product.VOLCANIC_ROCK_VOUCHER_9500] = (
                    coconut_coupon_take_orders + coconut_coupon_make_orders
                )
                # print(f"COCONUT_COUPON: {result[Product.COCONUT_COUPON]}")

            if coconut_orders != None:
                result[Product.VOLCANIC_ROCK] = coconut_orders
                # print(f"COCONUT: {result[Product.COCONUT]}")

        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData





