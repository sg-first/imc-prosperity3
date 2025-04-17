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
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET2 = "PICNIC_BASKET2"

# 为VOLCANIC_ROCK添加独立参数
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
        "volatility_mean": 0.18,
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
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 10,
    },
    Product.KELP: {
        "WINDOW_SIZE2": 200,
        "KELP_MEAN": 0,
        "buy_order_volume": 0,
        "sell_order_volume": 0,
    },
    Product.SQUID_INK: {"WINDOW_SIZE": 20, "BUY_THRESHOLD": 0.1, "SELL_THRESHOLD": 0.4},
    Product.JAMS: {
        "MAX_POS": 350,
    },
    Product.CROISSANTS: {
        "MAX_POS": 250,
    },
    Product.DJEMBES: {"MAX_POS": 60},
    Product.PICNIC_BASKET1: {"MAX_POS": 60},
    Product.PICNIC_BASKET2: {"MAX_POS": 100},
    Product.VOLCANIC_ROCK: {  # 新增VOLCANIC_ROCK参数
        "WINDOW_SIZE": 20,
        "BUY_THRESHOLD": 0.3,  # 买入阈值
        "SELL_THRESHOLD": 0.8,  # 卖出阈值
        "PRICE_LIMITS": {
            "max": 10500,  # 最高价
            "min": 9500,  # 最低价
        }
    }
}

LIMIT = {
    Product.VOLCANIC_ROCK: 400,
    **{voucher: 200 for voucher in Product.VOUCHERS}
}
BASKET_WEIGHTS = {Product.CROISSANTS: 6, Product.JAMS: 3, Product.DJEMBES: 1}

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
        if time_to_expiry <= 0:
            return max(0.0, (spot - strike) / spot)
        sigma = 0.2
        for _ in range(max_iterations):
            price = BlackScholes.black_scholes_call(spot, strike, time_to_expiry, sigma)
            vega = BlackScholes.vega(spot, strike, time_to_expiry, sigma)
            if abs(vega) < epsilon:
                sigma += 0.01
                continue
            diff = price - call_price
            if abs(diff) < tolerance:
                break
            sigma = sigma - diff / vega
            sigma = max(sigma, 0.01)
            sigma = min(sigma, 2.0)
        return sigma

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
    def __init__(self, max_position_change=50):
        self.risk_free_rate = 0.0
        self.max_position_change = max_position_change
        self.trader_data_template = {
            "volatility_history": {voucher: [] for voucher in Product.VOUCHERS},
            "underlying_history": []
        }
        self.trader_data = self.trader_data_template.copy()
        self.price_history = {}
        self.MAX_POS = 50
        self.WINDOW_SIZE = 20
        self.WINDOW_SIZE2 = 200
        self.KELP_MEAN = None
        self.BUY_THRESHOLD = 0.1
        self.SELL_THRESHOLD = 0.4
        self.PRICE_LIMITS = {
            "RAINFOREST_RESIN": {
                "max": 10003.5,
                "min": 9996.5,
            },
            "SQUID_INK": {
                "max": 2181.0,
                "min": 1740,
            },
            "KELP": {
                "max": 2050,
                "min": 2000,
            },
            "VOLCANIC_ROCK": {
                "max": 10500,
                "min": 9500,
            }
        }
        self.MRTraderObj = MRTrader(PARAMS)
        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            Product.DJEMBES: 60,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
        }
        self.spread_mean = 48.76
        self.spread_var = 85.12 ** 2
        self.halflife = 695
        self.lambda_ = 1 - np.exp(np.log(0.5) / self.halflife)
        self.components = {
            "CROISSANTS": 6,
            "JAMS": 3,
            "DJEMBES": 1
        }

    def update_spread_stats(self, spread: float):
        if not hasattr(self, "spread_mean"):
            self.spread_mean = spread
            self.spread_var = 0.0
            return 0.0
        new_mean = (1 - self.lambda_) * self.spread_mean + self.lambda_ * spread
        new_var = (1 - self.lambda_) * self.spread_var + self.lambda_ * (
                spread - new_mean
        ) ** 2
        self.spread_mean = new_mean
        self.spread_var = new_var
        return np.sqrt(new_var)

    def handle_basket_arbitrage1(self, state: TradingState):
        orders = []
        basket_symbol = "PICNIC_BASKET1"
        basket_mid = self.get_mid_price(state.order_depths[basket_symbol])
        component_mids = {
            p: self.get_mid_price(state.order_depths[p]) for p in self.components
        }
        basket_pos = state.position.get(basket_symbol, 0)
        fair_value = sum(qty * component_mids[p] for p, qty in self.components.items())
        spread = basket_mid - fair_value
        current_std = self.update_spread_stats(spread)
        z_score = (spread - self.spread_mean) / current_std if current_std > 1e-8 else 0.0

        def get_max_pos(symbol: str) -> int:
            return PARAMS[symbol]["MAX_POS"]

        if z_score > 2:
            basket_asks = state.order_depths[basket_symbol].sell_orders
            if not basket_asks:
                return orders
            best_ask = min(basket_asks.keys())
            max_sell = min(
                basket_asks[best_ask],
                get_max_pos(basket_symbol) + basket_pos,
            )
            component_limits = []
            for p, qty in self.components.items():
                current_p_pos = state.position.get(p, 0)
                max_p_buy = (get_max_pos(p) - current_p_pos) // qty
                component_limits.append(max_p_buy)
            final_volume = min(max_sell, *component_limits)
            if final_volume > 0:
                orders.append(Order(basket_symbol, best_ask, -final_volume))
                for p, qty in self.components.items():
                    p_asks = state.order_depths[p].sell_orders
                    if not p_asks:
                        continue
                    best_p_ask = min(p_asks.keys())
                    buy_volume = final_volume * qty
                    orders.append(Order(p, best_p_ask, buy_volume))

        elif z_score < -2:
            basket_bids = state.order_depths[basket_symbol].buy_orders
            if not basket_bids:
                return orders
            best_bid = max(basket_bids.keys())
            max_buy = min(
                basket_bids[best_bid],
                get_max_pos(basket_symbol) - basket_pos,
            )
            component_limits = []
            for p, qty in self.components.items():
                current_p_pos = state.position.get(p, 0)
                max_p_sell = (get_max_pos(p) + current_p_pos) // qty
                component_limits.append(max_p_sell)
            final_volume = min(max_buy, *component_limits)
            if final_volume > 0:
                orders.append(Order(basket_symbol, best_bid, final_volume))
                for p, qty in self.components.items():
                    p_bids = state.order_depths[p].buy_orders
                    if not p_bids:
                        continue
                    best_p_bid = max(p_bids.keys())
                    sell_volume = final_volume * qty
                    orders.append(Order(p, best_p_bid, -sell_volume))

        elif abs(z_score) < 0.2 and basket_pos != 0:
            close_volume = -basket_pos
            if close_volume > 0:
                best_bid = max(state.order_depths[basket_symbol].buy_orders.keys())
                orders.append(Order(basket_symbol, best_bid, close_volume))
            else:
                best_ask = min(state.order_depths[basket_symbol].sell_orders.keys())
                orders.append(Order(basket_symbol, best_ask, close_volume))
            for p, qty in self.components.items():
                p_pos = state.position.get(p, 0)
                if p_pos == 0:
                    continue
                hedge_volume = -p_pos * (abs(close_volume) / abs(basket_pos))
                hedge_volume = int(hedge_volume)
                if hedge_volume > 0:
                    best_bid = max(state.order_depths[p].buy_orders.keys())
                    orders.append(Order(p, best_bid, hedge_volume))
                else:
                    best_ask = min(state.order_depths[p].sell_orders.keys())
                    orders.append(Order(p, best_ask, hedge_volume))

        return orders

    def hedge_orders(self, state: TradingState, greeks: Dict) -> List[Order]:
        total_delta = sum(
            pos * greeks[voucher]["delta"]
            for voucher, pos in state.position.items()
            if voucher in greeks
        )
        target_position = -total_delta
        current_position = state.position.get(Product.VOLCANIC_ROCK, 0)
        delta_gap = target_position - current_position
        vol = np.std(self.trader_data["underlying_history"][-20:]) if len(
            self.trader_data["underlying_history"]) >= 20 else 0.2
        adjust_ratio = min(1.0, 0.5 / (vol + 0.01))
        adjusted_gap = delta_gap * adjust_ratio
        orders = []
        best_ask = min(state.order_depths[Product.VOLCANIC_ROCK].sell_orders.keys(), default=None)
        best_bid = max(state.order_depths[Product.VOLCANIC_ROCK].buy_orders.keys(), default=None)
        if adjusted_gap > 0 and best_ask:
            price = best_ask - 1
            qty = min(adjusted_gap, LIMIT[Product.VOLCANIC_ROCK] - current_position)
            orders.append(Order(Product.VOLCANIC_ROCK, price, qty))
        elif adjusted_gap < 0 and best_bid:
            price = best_bid + 1
            qty = max(adjusted_gap, -LIMIT[Product.VOLCANIC_ROCK] - current_position)
            orders.append(Order(Product.VOLCANIC_ROCK, price, qty))
        return orders

    def get_price_direction(self, prices: list) -> float:
        if len(prices) < self.WINDOW_SIZE:
            if len(prices) < 2:
                return 0
            else:
                return prices[-1] - prices[0]
        window = prices[-self.WINDOW_SIZE :]
        return window[-1] - window[0]

    def get_kelp_mean_price(self, product: str):
        if product in self.price_history and len(self.price_history[product]) > 0:
            if len(self.price_history[product]) > self.WINDOW_SIZE2:
                self.price_history[product].pop(0)
            self.KELP_MEAN = np.mean(self.price_history[product])

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
            return 0.0

    def calculate_greeks(self, state: TradingState) -> Dict[str, Dict[str, float]]:
        greeks = {}
        if Product.VOLCANIC_ROCK in state.order_depths:
            spot = self.get_mid_price(state.order_depths[Product.VOLCANIC_ROCK])
            self.trader_data["underlying_history"].append(spot)
            if len(self.trader_data["underlying_history"]) > 100:
                self.trader_data["underlying_history"] = self.trader_data["underlying_history"][-100:]
        else:
            spot = 0.0
        days_passed = state.timestamp / (1000 * 60 * 60 * 24)
        time_to_expiry = max((5 - days_passed) / 252, 1e-4)
        for voucher in Product.VOUCHERS:
            if voucher not in state.order_depths:
                continue
            strike = PARAMS[voucher]["strike"]
            voucher_price = self.get_mid_price(state.order_depths[voucher])
            if voucher_price <= 0 or spot <= 0:
                continue
            iv = BlackScholes.implied_volatility(voucher_price, spot, strike, time_to_expiry)
            self.trader_data["volatility_history"][voucher].append(iv)
            if len(self.trader_data["volatility_history"][voucher]) > PARAMS[voucher]["std_window"] * 2:
                self.trader_data["volatility_history"][voucher] = self.trader_data["volatility_history"][voucher][-PARAMS[voucher]["std_window"] * 2:]
            delta = BlackScholes.delta(spot, strike, time_to_expiry, iv)
            gamma = BlackScholes.gamma(spot, strike, time_to_expiry, iv)
            greeks[voucher] = {"delta": delta, "gamma": gamma, "iv": iv}
        spot_history = self.trader_data["underlying_history"]
        if len(spot_history) >= 20:
            returns = np.diff(np.log(spot_history[-20:]))
            gamma_risk = np.std(returns) * 100
            self.max_position_change = int(50 / (gamma_risk + 0.1))
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
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    sell_qty = min(LIMIT[voucher] + position, order_depth.buy_orders[best_bid])
                    if sell_qty > 0:
                        orders.append(Order(voucher, best_bid, -sell_qty))
            elif z_score < -params["zscore_threshold"]:
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    buy_qty = min(LIMIT[voucher] - position, -order_depth.sell_orders[best_ask])
                    if buy_qty > 0:
                        orders.append(Order(voucher, best_ask, buy_qty))
        return orders

    def run(self, state: TradingState):
        if state.traderData:
            self.trader_data.update(jsonpickle.decode(state.traderData))
        result = {}
        try:
            greeks = self.calculate_greeks(state)
            for voucher in Product.VOUCHERS:
                if voucher in state.order_depths:
                    orders = self.voucher_strategy(state, voucher)
                    if orders:
                        result[voucher] = orders
            for product in state.order_depths:
                if product in [Product.KELP, Product.SQUID_INK, Product.PICNIC_BASKET1,
                            Product.CROISSANTS, Product.JAMS, Product.DJEMBES,
                            Product.PICNIC_BASKET2, Product.RAINFOREST_RESIN, Product.VOLCANIC_ROCK]:
                    if product not in self.price_history:
                        self.price_history[product] = []
                    order_depth = state.order_depths[product]
                    orders: List[Order] = []
                    current_pos = state.position.get(product, 0)
                    mid_price = self.get_mid_price(order_depth)
                    self.price_history[product].append(mid_price)
                    if product == Product.RAINFOREST_RESIN:
                        orders = self.MRTraderObj.handle_resin_trading(state)
                    elif product == Product.PICNIC_BASKET1:
                        orders = self.handle_basket_arbitrage1(state)
                    elif product == Product.KELP:
                        self.get_kelp_mean_price(product)
                        if self.KELP_MEAN is not None and len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
                            best_ask = min(order_depth.sell_orders.keys())
                            best_bid = max(order_depth.buy_orders.keys())
                            if best_ask < self.KELP_MEAN and current_pos < self.MAX_POS:
                                ask_volume = abs(order_depth.sell_orders[best_ask])
                                buy_volume = min(ask_volume, self.MAX_POS - current_pos)
                                if buy_volume > 0:
                                    orders.append(Order(product, best_ask, buy_volume))
                            if best_bid > self.KELP_MEAN and current_pos > -self.MAX_POS:
                                bid_volume = abs(order_depth.buy_orders[best_bid])
                                sell_volume = min(bid_volume, self.MAX_POS + current_pos)
                                if sell_volume > 0:
                                    orders.append(Order(product, best_bid, -sell_volume))
                    elif product == Product.SQUID_INK:
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
                                    if mid_price < (max_price - min_price) * self.BUY_THRESHOLD + min_price and price_direction < 5 and current_pos < self.MAX_POS:
                                        ask_volume = abs(order_depth.sell_orders[best_ask])
                                        buy_volume = min(ask_volume, self.MAX_POS - current_pos, 5)
                                        if buy_volume > 0:
                                            orders.append(Order(product, best_ask, buy_volume))
                                    elif mid_price > (max_price - min_price) * self.SELL_THRESHOLD + min_price and price_direction > 5 and current_pos > -self.MAX_POS:
                                        bid_volume = abs(order_depth.buy_orders[best_bid])
                                        sell_volume = min(bid_volume, self.MAX_POS + current_pos)
                                        if sell_volume > 0:
                                            orders.append(Order(product, best_bid, -sell_volume))
                    elif product == Product.VOLCANIC_ROCK:  # 新增VOLCANIC_ROCK逻辑
                        if len(self.price_history[product]) > PARAMS[product]["WINDOW_SIZE"]:
                            self.price_history[product].pop(0)
                        if len(self.price_history[product]) >= 2:
                            price_direction = self.get_price_direction(self.price_history[product])
                            price_limits = PARAMS[product]["PRICE_LIMITS"]
                            max_price = price_limits["max"]
                            min_price = price_limits["min"]
                            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
                            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0
                            if best_bid and best_ask:
                                if mid_price < (max_price - min_price) * PARAMS[product]["BUY_THRESHOLD"] + min_price and price_direction < 5 and current_pos < self.MAX_POS:
                                    ask_volume = abs(order_depth.sell_orders[best_ask])
                                    buy_volume = min(ask_volume, self.MAX_POS - current_pos, 5)
                                    if buy_volume > 0:
                                        orders.append(Order(product, best_ask, buy_volume))
                                elif mid_price > (max_price - min_price) * PARAMS[product]["SELL_THRESHOLD"] + min_price and price_direction > 5 and current_pos > -self.MAX_POS:
                                    bid_volume = abs(order_depth.buy_orders[best_bid])
                                    sell_volume = min(bid_volume, self.MAX_POS + current_pos)
                                    if sell_volume > 0:
                                        orders.append(Order(product, best_bid, -sell_volume))
                    if orders:
                        result[product] = orders
        except Exception as e:
            print(f"Error: {str(e)}")
        return result, 0, jsonpickle.encode(self.trader_data)

class MRTrader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
        }

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
                    quantity = min(best_ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
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
            orders.append(Order(product, round(bid), buy_quantity))
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))
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
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
        if position_after_take < 0:
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
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

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        manage_position: bool = False,
        soft_position_limit: int = 0,
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
                ask = best_ask_above_fair
            else:
                ask = best_ask_above_fair - 1
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

    def handle_resin_trading(trader, state: TradingState):
        resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
        order_depth = state.order_depths[Product.RAINFOREST_RESIN]
        resin_take_orders, buy_order_volume, sell_order_volume = trader.take_orders(
            Product.RAINFOREST_RESIN,
            state.order_depths[Product.RAINFOREST_RESIN],
            trader.params[Product.RAINFOREST_RESIN]["fair_value"],
            trader.params[Product.RAINFOREST_RESIN]["take_width"],
            resin_position,
        )
        resin_clear_orders, buy_order_volume, sell_order_volume = trader.clear_orders(
            Product.RAINFOREST_RESIN,
            state.order_depths[Product.RAINFOREST_RESIN],
            trader.params[Product.RAINFOREST_RESIN]["fair_value"],
            trader.params[Product.RAINFOREST_RESIN]["clear_width"],
            resin_position,
            buy_order_volume,
            sell_order_volume,
        )
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
        return resin_take_orders + resin_clear_orders + resin_make_orders

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        result = {}
        if (
            Product.RAINFOREST_RESIN in self.params
            and Product.RAINFOREST_RESIN in state.order_depths
        ):
            result[Product.RAINFOREST_RESIN] = self.handle_resin_trading(self, state)
        conversions = 1
        traderData = jsonpickle.encode(traderObject)
        return result, conversions, traderData