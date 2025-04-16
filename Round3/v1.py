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
    VOUCHERS = [
        "VOLCANIC_ROCK_VOUCHER_9500",
        "VOLCANIC_ROCK_VOUCHER_9750",
        "VOLCANIC_ROCK_VOUCHER_10000",
        "VOLCANIC_ROCK_VOUCHER_10250",
        "VOLCANIC_ROCK_VOUCHER_10500",
    ]


PARAMS = {
    "VOLCANIC_ROCK_VOUCHER_9500": {
        "strike": 9500,
        "std_window": 20,
        "zscore_threshold": 1.5,
        "volatility_mean": 0.19,
    },
    "VOLCANIC_ROCK_VOUCHER_9750": {
        "strike": 9750,
        "std_window": 20,
        "zscore_threshold": 3,
        "volatility_mean": 0.18,#???-4w
    },
    "VOLCANIC_ROCK_VOUCHER_10000": {
        "strike": 10000,
        "std_window": 20,
        "zscore_threshold": 1.5,
        "volatility_mean": 0.16,
    },
    "VOLCANIC_ROCK_VOUCHER_10250": {
        "strike": 10250,
        "std_window": 20,
        "zscore_threshold": 1.5,
        "volatility_mean": 0.15,
    },
    "VOLCANIC_ROCK_VOUCHER_10500": {
        "strike": 10500,
        "std_window": 20,
        "zscore_threshold": 1.5,
        "volatility_mean": 0.15,
    },
    "VOLCANIC_ROCK": {
        "position_limit": 400,
    }
}

LIMIT = {
    Product.VOLCANIC_ROCK: 400,
    **{voucher: 200 for voucher in Product.VOUCHERS}
}


class BlackScholes:
    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.01
        high_vol = 2.0  # 扩大波动率范围
        volatility = (low_vol + high_vol) / 2.0
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility

    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        if time_to_expiry <= 0:
            return max(0.0, spot - strike)
        d1 = (log(spot / strike) + (0.5 * volatility ** 2) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        return spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        if time_to_expiry <= 0:
            return 1.0 if spot > strike else 0.0
        d1 = (log(spot / strike) + (0.5 * volatility ** 2) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def gamma(spot, strike, time_to_expiry, volatility):
        if time_to_expiry <= 0:
            return 0.0
        d1 = (log(spot / strike) + (0.5 * volatility ** 2) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) / (spot * volatility * sqrt(time_to_expiry))


class Trader:
    def __init__(self):
        self.risk_free_rate = 0.0
        self.trader_data_template = {
            "volatility_history": {voucher: [] for voucher in Product.VOUCHERS},
            "underlying_history": []
        }
        self.trader_data = self.trader_data_template.copy()

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0
        if best_bid and best_ask:
            return (best_bid + best_ask) / 2
        elif best_bid:
            return best_bid
        elif best_ask:
            return best_ask
        else:
            return 0.0  # 默认值，需根据实际情况处理

    def calculate_greeks(self, state: TradingState) -> Dict[str, Dict[str, float]]:
        greeks = {}
        if Product.VOLCANIC_ROCK in state.order_depths:
            spot = self.get_mid_price(state.order_depths[Product.VOLCANIC_ROCK])
            self.trader_data["underlying_history"].append(spot)
            # 保留最近100个价格数据
            if len(self.trader_data["underlying_history"]) > 100:
                self.trader_data["underlying_history"] = self.trader_data["underlying_history"][-100:]
        else:
            spot = 0.0

        time_to_expiry = ( 5/252- (state.timestamp) / 1000000 / 252)  # 处理到期时间非负

        for voucher in Product.VOUCHERS:
            if voucher not in state.order_depths:
                continue

            strike = PARAMS[voucher]["strike"]
            voucher_price = self.get_mid_price(state.order_depths[voucher])
            if voucher_price <= 0 or spot <= 0:
                continue

            iv = BlackScholes.implied_volatility(voucher_price, spot, strike, time_to_expiry)
            self.trader_data["volatility_history"][voucher].append(iv)
            # 保留最近std_window * 2的数据
            if len(self.trader_data["volatility_history"][voucher]) > PARAMS[voucher]["std_window"] * 2:
                self.trader_data["volatility_history"][voucher] = self.trader_data["volatility_history"][voucher][-PARAMS[voucher]["std_window"] * 2:]

            delta = BlackScholes.delta(spot, strike, time_to_expiry, iv)
            gamma = BlackScholes.gamma(spot, strike, time_to_expiry, iv)
            greeks[voucher] = {"delta": delta, "gamma": gamma, "iv": iv}

        return greeks

    def hedge_orders(self, state: TradingState, greeks: Dict) -> List[Order]:
        orders = []
        if Product.VOLCANIC_ROCK not in state.order_depths:
            return orders

        current_position = state.position.get(Product.VOLCANIC_ROCK, 0)
        total_delta = sum(
            pos * greeks[voucher]["delta"]
            for voucher, pos in state.position.items()
            if voucher in greeks
        )
        target_position = -int(total_delta)
        quantity = target_position - current_position

        if quantity != 0:
            order_depth = state.order_depths[Product.VOLCANIC_ROCK]
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None

            if quantity > 0:
                if not best_ask:
                    return orders
                qty = min(quantity, LIMIT[Product.VOLCANIC_ROCK] - current_position)
                orders.append(Order(Product.VOLCANIC_ROCK, best_ask, qty))
            else:
                if not best_bid:
                    return orders
                qty = max(quantity, -LIMIT[Product.VOLCANIC_ROCK] - current_position)
                orders.append(Order(Product.VOLCANIC_ROCK, best_bid, qty))

        return orders

    def voucher_strategy(self, state: TradingState, voucher: str) -> List[Order]:
        params = PARAMS[voucher]
        orders = []
        if voucher not in state.order_depths:
            return orders

        order_depth = state.order_depths[voucher]
        position = state.position.get(voucher, 0)
        volatility_history = self.trader_data["volatility_history"][voucher]

        if len(volatility_history) >= params["std_window"]:
            current_vol = volatility_history[-1]
            window = volatility_history[-params["std_window"]:]
            vol_mean = np.mean(window)
            vol_std = np.std(window)
            if vol_std == 0:
                return orders

            z_score = (current_vol - vol_mean) / vol_std
            if z_score > params["zscore_threshold"]:
                # 做空波动率：卖出期权
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    sell_qty = min(LIMIT[voucher] + position, order_depth.buy_orders[best_bid])
                    if sell_qty > 0:
                        orders.append(Order(voucher, best_bid, -sell_qty))
            elif z_score < -params["zscore_threshold"]:
                # 做多波动率：买入期权
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    buy_qty = min(LIMIT[voucher] - position, -order_depth.sell_orders[best_ask])
                    if buy_qty > 0:
                        orders.append(Order(voucher, best_ask, buy_qty))

        return orders

    def run(self, state: TradingState):
        # 初始化trader_data
        if state.traderData:
            self.trader_data = jsonpickle.decode(state.traderData)
        else:
            self.trader_data = self.trader_data_template.copy()

        result = {}
        try:
            # 计算希腊值并更新历史数据
            greeks = self.calculate_greeks(state)
            # 对冲标的仓位
            rock_orders = self.hedge_orders(state, greeks)
            if rock_orders:
                result[Product.VOLCANIC_ROCK] = rock_orders
            # 执行期权策略
            for voucher in Product.VOUCHERS:
                if voucher in state.order_depths:
                    voucher_orders = self.voucher_strategy(state, voucher)
                    if voucher_orders:
                        result[voucher] = voucher_orders
        except Exception as e:
            print(f"Error: {e}")

        return result, 0, jsonpickle.encode(self.trader_data)