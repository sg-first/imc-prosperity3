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
        "zscore_threshold": 5,
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
    def vega(spot, strike, time_to_expiry, volatility):
        if time_to_expiry <= 0:
            return 0.0
        d1 = (log(spot / strike) + (0.5 * volatility ** 2) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return spot * sqrt(time_to_expiry) * NormalDist().pdf(d1)

    @staticmethod
    def implied_volatility(
            call_price, spot, strike, time_to_expiry,
            max_iterations=100, tolerance=1e-8, epsilon=1e-8
    ):
        # 处理极端情况
        if time_to_expiry <= 0:
            return max(0.0, (spot - strike) / spot)  # 返回内在价值对应的波动率

        # 初始猜测值
        sigma = 0.2

        for _ in range(max_iterations):
            price = BlackScholes.black_scholes_call(spot, strike, time_to_expiry, sigma)
            vega = BlackScholes.vega(spot, strike, time_to_expiry, sigma)

            # 防止除零错误
            if abs(vega) < epsilon:
                sigma += 0.01  # 轻微扰动
                continue

            diff = price - call_price
            if abs(diff) < tolerance:
                break

            # 牛顿迭代公式
            sigma = sigma - diff / vega

            # 波动率范围保护
            sigma = max(sigma, 0.01)
            sigma = min(sigma, 2.0)

        return sigma

    # 保持其他方法不变（black_scholes_call, delta, gamma）
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
    def __init__(self, max_position_change=50):  # 设置最大头寸变动值的默认参数
        self.risk_free_rate = 0.0
        self.max_position_change = max_position_change  # 最大允许的头寸变动值
        self.trader_data_template = {
            "volatility_history": {voucher: [] for voucher in Product.VOUCHERS},
            "underlying_history": []
        }
        self.trader_data = self.trader_data_template.copy()

    def hedge_orders(self, state: TradingState, greeks: Dict) -> List[Order]:
        # 计算总Delta时保留浮点数精度
        total_delta = sum(
            pos * greeks[voucher]["delta"]
            for voucher, pos in state.position.items()
            if voucher in greeks
        )

        # 目标仓位保留浮点，按比例调整
        target_position = -total_delta
        current_position = state.position.get(Product.VOLCANIC_ROCK, 0)
        delta_gap = target_position - current_position

        # 动态调整对冲比例（根据市场波动率）
        vol = np.std(self.trader_data["underlying_history"][-20:]) if len(
            self.trader_data["underlying_history"]) >= 20 else 0.2
        adjust_ratio = min(1.0, 0.5 / (vol + 0.01))  # 波动率越高，对冲比例越低
        adjusted_gap = delta_gap * adjust_ratio

        # 生成订单（限价单挂单）
        orders = []
        best_ask = min(state.order_depths[Product.VOLCANIC_ROCK].sell_orders.keys(), default=None)
        best_bid = max(state.order_depths[Product.VOLCANIC_ROCK].buy_orders.keys(), default=None)

        if adjusted_gap > 0 and best_ask:
            price = best_ask - 1  # 挂单在买一价下方1单位
            qty = min(adjusted_gap, LIMIT[Product.VOLCANIC_ROCK] - current_position)
            orders.append(Order(Product.VOLCANIC_ROCK, price, qty))
        elif adjusted_gap < 0 and best_bid:
            price = best_bid + 1  # 挂单在卖一价上方1单位
            qty = max(adjusted_gap, -LIMIT[Product.VOLCANIC_ROCK] - current_position)
            orders.append(Order(Product.VOLCANIC_ROCK, price, qty))

        return orders

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

        # 修正后的时间计算（假设timestamp单位是毫秒）
        days_passed = state.timestamp / (1000 * 60 * 60 * 24)  # 转换为天数
        time_to_expiry = max((5 - days_passed) / 252, 1e-4)  # 年化，最小保留0.0001防止除零

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

        spot_history = self.trader_data["underlying_history"]
        if len(spot_history) >= 20:
            returns = np.diff(np.log(spot_history[-20:]))
            gamma_risk = np.std(returns) * 100  # 波动率放大100倍作为风险系数
            self.max_position_change = int(50 / (gamma_risk + 0.1))  # 波动率高时减少单次对冲量
        else:
            self.max_position_change = 50

        return greeks




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
        # 初始化数据
        if state.traderData:
            self.trader_data.update(jsonpickle.decode(state.traderData))

        result = {}
        try:
            # 计算希腊值
            greeks = self.calculate_greeks(state)

            #执行对冲
            # hedge_orders = self.hedge_orders(state, greeks)
            # if hedge_orders:
            #     result[Product.VOLCANIC_ROCK] = hedge_orders

            # 期权波动率策略
            for voucher in Product.VOUCHERS:
                if voucher in state.order_depths:
                    orders = self.voucher_strategy(state, voucher)
                    if orders:
                        result[voucher] = orders



        except Exception as e:
            print(f"Error: {str(e)}")

        return result, 0, jsonpickle.encode(self.trader_data)