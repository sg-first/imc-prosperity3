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
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"


PARAMS = {
Product.MAGNIFICENT_MACARONS:{
        "make_edge": 1.8,          # 初始报价边缘
        "make_min_edge": 1.3,       # 最小允许边缘
        "init_make_edge": 1.8,      # 初始边缘
        "volume_avg_timestamp":5,# 成交量计算窗口
        "volume_bar":20,          # 成交量调整阈值
        "dec_edge_discount":0.8,  # 减价折扣系数
        "step_size":0.2           # 边缘调整步长
    },
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
        "volatility_mean": 0.18,  # ???-4w
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
    },
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,  # 设定商品的基准价格
        "take_width": 1,  # 允许吃单的价差阈值,当对手方订单价格与fair_value的偏离超过clear_width时吃单
        "clear_width": 0,  # 平仓订单的挂单价差范围,要平仓时在fair_value ± clear_width挂单。
        # for making
        "disregard_edge": 1,  # 忽略对手方订单的范围阈值，挂单时忽略对手方订单中与fair_value 过近的订单
        "join_edge": 2,  # 加入已有订单的价差容忍范围，当订单价格在fair_value ± join_edge内时挂相同价格的订单，否则挂更优价格。
        "default_edge": 4,  # 默认挂单价差,当没有可直接交易的订单时，以fair_value ± default_edge挂单。
        "soft_position_limit": 10,
    },
    Product.KELP: {
        "WINDOW_SIZE2": 200,
        "KELP_MEAN": 0,
        "buy_order_volume": 0,
        "sell_order_volume": 0,
    },
    Product.SQUID_INK: {"WINDOW_SIZE": 20, "BUY_THRESHOLD": 0.4, "SELL_THRESHOLD": 0.7},
    Product.JAMS: {
        "MAX_POS": 350,
    },
    Product.CROISSANTS: {
        "MAX_POS": 250,
    },
    Product.DJEMBES: {"MAX_POS": 60},
    Product.PICNIC_BASKET1: {"MAX_POS": 60},
    Product.PICNIC_BASKET2: {"MAX_POS": 100},


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
    def __init__(self, max_position_change=50,params = None):  # 设置最大头寸变动值的默认参数
        if params is None:
            params = PARAMS
        self.params = params
        self.sunlight_history = []  # 新增阳光指数历史数据存储
        self.TREND_WINDOW = 10  # 趋势分析窗口大小
        self.THRESHOLD = 0.1  # 趋势触发阈值

        self.risk_free_rate = 0.0
        self.max_position_change = max_position_change  # 最大允许的头寸变动值
        self.trader_data_template = {
            "volatility_history": {voucher: [] for voucher in Product.VOUCHERS},
            "underlying_history": []
        }
        self.trader_data = self.trader_data_template.copy()
        self.price_history = {}  # 存储每个产品的价格历史
        # 不需要自己维护position了，因为TradingState中已经有了
        self.MAX_POS = 50
        self.WINDOW_SIZE = 20  # 滑动窗口大小
        self.WINDOW_SIZE2 = 200
        self.KELP_MEAN = None
        self.BUY_THRESHOLD = 0.1  # 买入阈值：低于最低价+极差的20%
        self.SELL_THRESHOLD = 0.4  # 卖出阈值：高于最低价+极差的80%
        self.PRICE_LIMITS = {  # 为不同产品设置固定的极值点参数
            "RAINFOREST_RESIN": {
                "max": 10003.5,  # 最高价
                "min": 9996.5,  # 最低价
            },
            "SQUID_INK": {
                "max": 2181.0,  # 最高价
                "min": 1740,  # 最低价
            },
            "KELP": {
                "max": 2050,  # 最高价
                "min": 2000,  # 最低价
            },
            "VOLCANIC_ROCK":{
                "max": 10500, #最高价
                "min": 9500, #最低价
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
            Product.MAGNIFICENT_MACARONS: 75
        }
        self.spread_mean = 48.76  # 初始均值
        self.spread_var = 85.12 ** 2  # 初始方差
        self.halflife = 695  # 半衰期
        self.lambda_ = 1 - np.exp(np.log(0.5) / self.halflife)  # 衰减因子
        self.components = {
            "CROISSANTS": 6,
            "JAMS": 3,
            "DJEMBES": 1
        }

    def calculate_sunlight_trend(self) -> float:
        """计算加权阳光指数趋势"""
        if len(self.sunlight_history) < 2:
            return 0

        # 使用指数加权移动平均
        weights = np.exp(np.linspace(0, 1, len(self.sunlight_history)))
        weights /= weights.sum()
        diff = np.diff(self.sunlight_history[-self.TREND_WINDOW:])
        return np.dot(diff, weights[:len(diff)])

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

    def MAGNIFICENT_MACARONS_arb_take(self,
            order_depth: OrderDepth,
            observation: ConversionObservation,
            adap_edge: float,
            position: int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        # 更新阳光指数数据
        self.sunlight_history.append(observation.sunlightIndex)
        if len(self.sunlight_history) > self.TREND_WINDOW * 2:
            self.sunlight_history = self.sunlight_history[-self.TREND_WINDOW * 2:]

        # 计算趋势（正值表示上涨，负值表示下跌）
        trend = self.calculate_sunlight_trend()
        print(f"[DEBUG] Sunlight Trend: {trend:.2f}")

        # 动态计算参考价格
        best_bid = max(order_depth.buy_orders.keys(), default=0)
        best_ask = min(order_depth.sell_orders.keys(), default=0)
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0

        # 修正趋势逻辑 --------------------------------------------------
        if trend <- self.THRESHOLD:  # 阳光上涨->看空->卖出
            max_sell = self.LIMIT[Product.MAGNIFICENT_MACARONS] + position
            print(f"[SELL] Max allowed: {max_sell}")

            # 从最优买价开始吃单
            for price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if sell_order_volume >= max_sell:
                    break
                # 移除固定价格过滤
                available_qty = order_depth.buy_orders[price]
                execute_qty = min(available_qty, max_sell - sell_order_volume)
                if execute_qty > 0:
                    orders.append(Order(Product.MAGNIFICENT_MACARONS, price, -execute_qty))
                    sell_order_volume += execute_qty
                    print(f"[SELL] Executed {execute_qty} @ {price}")

        elif trend > self.THRESHOLD:  # 阳光下跌->看多->买入
            max_buy = self.LIMIT[Product.MAGNIFICENT_MACARONS] - position
            print(f"[BUY] Max allowed: {max_buy}")

            # 从最优卖价开始吃单
            for price in sorted(order_depth.sell_orders.keys()):
                if buy_order_volume >= max_buy:
                    break
                # 移除固定价格过滤
                available_qty = abs(order_depth.sell_orders[price])
                execute_qty = min(available_qty, max_buy - buy_order_volume)
                if execute_qty > 0:
                    orders.append(Order(Product.MAGNIFICENT_MACARONS, price, execute_qty))
                    buy_order_volume += execute_qty
                    print(f"[BUY] Executed {execute_qty} @ {price}")

        return orders, buy_order_volume, sell_order_volume

    def MAGNIFICENT_MACARONS_arb_clear(
            self,
            position: int
    ) -> int:
        conversions = -position

        conversions = max(-10, min(10, conversions))
        return conversions

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
        aggressive_ask = foreign_mid - 1.6
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

    def update_spread_stats(self, spread: float):
        """指数加权更新价差统计量"""
        # 初始状态处理
        if not hasattr(self, "spread_mean"):
            # 首次调用时初始化
            self.spread_mean = spread
            self.spread_var = 0.0
            return 0.0  # 初始标准差为0

        # 更新均值
        new_mean = (1 - self.lambda_) * self.spread_mean + self.lambda_ * spread

        # 更新方差（使用新均值！）
        new_var = (1 - self.lambda_) * self.spread_var + self.lambda_ * (
                spread - new_mean
        ) ** 2

        # 保存状态
        self.spread_mean = new_mean
        self.spread_var = new_var

        return np.sqrt(new_var)  # 返回标准差

    def handle_basket_arbitrage1(self, state: TradingState):
        """处理野餐篮套利逻辑，严格匹配成分交易"""
        orders = []
        basket_symbol = "PICNIC_BASKET1"

        # 获取中间价和仓位
        basket_mid = self.get_mid_price(state.order_depths[basket_symbol])
        component_mids = {
            p: self.get_mid_price(state.order_depths[p]) for p in self.components
        }
        basket_pos = state.position.get(basket_symbol, 0)

        # 计算价差和Z值
        fair_value = sum(qty * component_mids[p] for p, qty in self.components.items())
        spread = basket_mid - fair_value

        current_std = self.update_spread_stats(spread)
        z_score = (spread - self.spread_mean) / current_std if current_std > 1e-8 else 0.0

        def get_max_pos(symbol: str) -> int:
            return PARAMS[symbol]["MAX_POS"]

        # 做空价差：卖篮子 + 买成分
        if z_score > 2:
            # 计算最大可卖出篮子数量
            basket_asks = state.order_depths[basket_symbol].sell_orders
            if not basket_asks:
                return orders

            best_ask = min(basket_asks.keys())
            max_sell = min(
                basket_asks[best_ask],
                get_max_pos(basket_symbol) + basket_pos,  # 当前空头仓位限制
            )

            # 计算成分限制
            component_limits = []
            for p, qty in self.components.items():
                current_p_pos = state.position.get(p, 0)
                max_p_buy = (get_max_pos(p) - current_p_pos) // qty  # 确保整数倍
                component_limits.append(max_p_buy)

            final_volume = min(max_sell, *component_limits)

            if final_volume > 0:
                # 卖出篮子
                orders.append(Order(basket_symbol, best_ask, -final_volume))

                # 买入成分（吃对手方的卖单）
                for p, qty in self.components.items():
                    p_asks = state.order_depths[p].sell_orders
                    if not p_asks:
                        continue

                    best_p_ask = min(p_asks.keys())
                    buy_volume = final_volume * qty
                    orders.append(Order(p, best_p_ask, buy_volume))

        # 做多价差：买篮子 + 卖成分
        elif z_score < -2:
            basket_bids = state.order_depths[basket_symbol].buy_orders
            if not basket_bids:
                return orders

            best_bid = max(basket_bids.keys())
            max_buy = min(
                basket_bids[best_bid],
                get_max_pos(basket_symbol) - basket_pos,  # 当前多头仓位限制
            )

            # 计算成分限制
            component_limits = []
            for p, qty in self.components.items():
                current_p_pos = state.position.get(p, 0)
                max_p_sell = (get_max_pos(p) + current_p_pos) // qty  # 确保整数倍
                component_limits.append(max_p_sell)

            final_volume = min(max_buy, *component_limits)

            if final_volume > 0:
                # 买入篮子
                orders.append(Order(basket_symbol, best_bid, final_volume))

                # 卖出成分（吃对手方的买单）
                for p, qty in self.components.items():
                    p_bids = state.order_depths[p].buy_orders
                    if not p_bids:
                        continue

                    best_p_bid = max(p_bids.keys())
                    sell_volume = final_volume * qty
                    orders.append(Order(p, best_p_bid, -sell_volume))

        # 平仓逻辑（|Z|<0.5时平仓）
        elif abs(z_score) < 0.2 and basket_pos != 0:
            # 平仓篮子
            close_volume = -basket_pos
            if close_volume > 0:
                best_bid = max(state.order_depths[basket_symbol].buy_orders.keys())
                orders.append(Order(basket_symbol, best_bid, close_volume))
            else:
                best_ask = min(state.order_depths[basket_symbol].sell_orders.keys())
                orders.append(Order(basket_symbol, best_ask, close_volume))

            # 平仓成分
            for p, qty in self.components.items():
                p_pos = state.position.get(p, 0)
                if p_pos == 0:
                    continue

                # 计算需要平仓的数量（根据篮子平仓量比例）
                hedge_volume = -p_pos * (abs(close_volume) / abs(basket_pos))
                hedge_volume = int(hedge_volume)  # 必须为整数

                if hedge_volume > 0:
                    best_bid = max(state.order_depths[p].buy_orders.keys())
                    orders.append(Order(p, best_bid, hedge_volume))
                else:
                    best_ask = min(state.order_depths[p].sell_orders.keys())
                    orders.append(Order(p, best_ask, hedge_volume))

        return orders

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

    def get_price_direction(self, prices: list) -> float:
        """计算价格方向
        使用窗口内第一个点和最后一个点计算整体趋势斜率
        """
        if len(prices) < self.WINDOW_SIZE:  # 数据不足一个完整窗口
            if len(prices) < 2:  # 至少需要两个点
                return 0
            else:  # 使用可用数据的首尾两点
                return prices[-1] - prices[0]

        # 使用窗口内的首尾两点计算趋势
        window = prices[-self.WINDOW_SIZE:]  # 取最近的WINDOW_SIZE个价格
        return window[-1] - window[0]  # 终点减起点得到趋势

    def get_kelp_mean_price(self, product: str):
        if product in self.price_history and len(self.price_history[product]) > 0:
            if len(self.price_history[product]) > self.WINDOW_SIZE2:
                self.price_history[product].pop(0)  # 移除最旧的数据
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
                self.trader_data["volatility_history"][voucher] = self.trader_data["volatility_history"][voucher][
                                                                  -PARAMS[voucher]["std_window"] * 2:]

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

    def run_makalong(self, state: TradingState):
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
            MAGNIFICENT_MACARONS_position = (
                state.position[Product.MAGNIFICENT_MACARONS]
                if Product.MAGNIFICENT_MACARONS in state.position
                else 0
            )
            print(f"MAGNIFICENT_MACARONS POSITION: {MAGNIFICENT_MACARONS_position}")

            conversions = self.MAGNIFICENT_MACARONS_arb_clear(
                MAGNIFICENT_MACARONS_position
            )

            MAGNIFICENT_MACARONS_position += conversions

            if "sunlight_history" not in traderObject:
                traderObject["sunlight_history"] = []
            traderObject["sunlight_history"] = self.sunlight_history[-self.TREND_WINDOW * 2:]

            MAGNIFICENT_MACARONS_take_orders, buy_order_volume, sell_order_volume = self.MAGNIFICENT_MACARONS_arb_take(
                state.order_depths[Product.MAGNIFICENT_MACARONS],
                state.observations.conversionObservations[Product.MAGNIFICENT_MACARONS],
                traderObject["MAGNIFICENT_MACARONS"]["curr_edge"],
                MAGNIFICENT_MACARONS_position,
            )

            MAGNIFICENT_MACARONS_make_orders, _, _ = self.MAGNIFICENT_MACARONS_arb_make(
                state.order_depths[Product.MAGNIFICENT_MACARONS],
                state.observations.conversionObservations[Product.MAGNIFICENT_MACARONS],
                MAGNIFICENT_MACARONS_position,
                traderObject["MAGNIFICENT_MACARONS"]["curr_edge"],
                buy_order_volume,
                sell_order_volume
            )

            result[Product.MAGNIFICENT_MACARONS] = (
                    MAGNIFICENT_MACARONS_take_orders + MAGNIFICENT_MACARONS_make_orders
            )

        traderData = jsonpickle.encode(traderObject)

        return result[Product.MAGNIFICENT_MACARONS], conversions

    def run(self, state: TradingState):
        # 初始化数据
        if state.traderData:
            self.trader_data.update(jsonpickle.decode(state.traderData))

        conversions = 0
        result = {}
        try:
            # 计算希腊值
            greeks = self.calculate_greeks(state)

            # 执行对冲
            # hedge_orders = self.hedge_orders(state, greeks)
            # if hedge_orders:
            #     result[Product.VOLCANIC_ROCK] = hedge_orders

            # 期权波动率策略
            for voucher in Product.VOUCHERS:
                if voucher in state.order_depths:
                    orders = self.voucher_strategy(state, voucher)
                    if orders:
                        result[voucher] = orders






            # Round 2 的策略
            for product in state.order_depths:
                if product in [Product.KELP, Product.SQUID_INK, Product.PICNIC_BASKET1,
                               Product.CROISSANTS, Product.JAMS, Product.DJEMBES,
                               Product.PICNIC_BASKET2, Product.RAINFOREST_RESIN,Product.VOLCANIC_ROCK,Product.MAGNIFICENT_MACARONS]:

                    if product not in self.price_history:
                        self.price_history[product] = []

                    order_depth = state.order_depths[product]
                    orders: List[Order] = []
                    current_pos = state.position.get(product, 0)

                    # 计算当前中间价并更新价格历史
                    mid_price = self.get_mid_price(order_depth)
                    self.price_history[product].append(mid_price)

                    if product == Product.RAINFOREST_RESIN:
                        orders = self.MRTraderObj.handle_resin_trading(state)
                    elif product == Product.PICNIC_BASKET1:
                        orders = self.handle_basket_arbitrage1(state)
                    elif product == Product.MAGNIFICENT_MACARONS:
                        orders,conversions = self.run_makalong(state)
                    elif product == Product.KELP:
                        # KELP专用逻辑
                        self.get_kelp_mean_price(product)
                        if (
                                self.KELP_MEAN is not None
                                and len(order_depth.sell_orders) > 0
                                and len(order_depth.buy_orders) > 0
                        ):
                            best_ask = min(order_depth.sell_orders.keys())
                            best_bid = max(order_depth.buy_orders.keys())

                            if best_ask < self.KELP_MEAN and current_pos < self.MAX_POS:
                                ask_volume = abs(order_depth.sell_orders[best_ask])
                                buy_volume = min(ask_volume, self.MAX_POS - current_pos)
                                if buy_volume > 0:
                                    orders.append(Order(product, best_ask, buy_volume))
                                    # print(
                                    #     f"KELP BUY {buy_volume} @ {best_ask} (Mean: {self.KELP_MEAN:.2f})"
                                    # )

                        if best_bid > self.KELP_MEAN and current_pos > -self.MAX_POS:
                            bid_volume = abs(order_depth.buy_orders[best_bid])
                            sell_volume = min(bid_volume, self.MAX_POS + current_pos)
                            if sell_volume > 0:
                                orders.append(Order(product, best_bid, -sell_volume))
                                # print(
                                # f"KELP SELL {sell_volume} @ {best_bid} (Mean: {self.KELP_MEAN:.2f})"
                            # )
                    elif product == Product.SQUID_INK  :
                        # ink产品逻辑
                        if product == Product.SQUID_INK:
                            PARAMS = self.PRICE_LIMITS["SQUID_INK"]
                        else:
                            PARAMS = self.PRICE_LIMITS["VOLCANIC_ROCK"]
                        if len(self.price_history[product]) > self.WINDOW_SIZE:
                            self.price_history[product].pop(0)

                        if len(self.price_history[product]) >= 2:
                            price_direction = self.get_price_direction(
                                self.price_history[product]
                            )
                            price_limits = self.PRICE_LIMITS.get(product)

                            if price_limits:
                                max_price = price_limits["max"]
                                min_price = price_limits["min"]
                                best_bid = (
                                    max(order_depth.buy_orders.keys())
                                    if order_depth.buy_orders
                                    else 0
                                )
                                best_ask = (
                                    min(order_depth.sell_orders.keys())
                                    if order_depth.sell_orders
                                    else 0
                                )

                                if best_bid and best_ask:
                                    if (
                                            mid_price
                                            < (max_price - min_price) * self.BUY_THRESHOLD
                                            + min_price
                                            and price_direction < 5
                                            and current_pos < self.MAX_POS
                                    ):
                                        ask_volume = abs(order_depth.sell_orders[best_ask])
                                        buy_volume = min(
                                            ask_volume, self.MAX_POS - current_pos, 5
                                        )
                                        if buy_volume > 0:
                                            orders.append(Order(product, best_ask, buy_volume))
                                            print(
                                                f"{product} BUY {buy_volume} @ {best_ask} Direction: {price_direction:.2f}"
                                            )

                                    elif (
                                            mid_price
                                            > (max_price - min_price) * self.SELL_THRESHOLD
                                            + min_price
                                            and price_direction > 5
                                            and current_pos > -self.MAX_POS
                                    ):
                                        bid_volume = abs(order_depth.buy_orders[best_bid])
                                        sell_volume = min(
                                            bid_volume, self.MAX_POS + current_pos
                                        )
                                        if sell_volume > 0:
                                            orders.append(
                                                Order(product, best_bid, -sell_volume)
                                            )
                                            print(
                                                f"{product} SELL {sell_volume} @ {best_bid} Direction: {price_direction:.2f}"
                                            )
                    if orders:
                        result[product] = orders
        except Exception as e:
            print(f"Error: {str(e)}")

        return result, conversions, jsonpickle.encode(self.trader_data)


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

    def handle_resin_trading(trader, state: TradingState):
        # 从state获取当前仓位和订单簿
        resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
        order_depth = state.order_depths[Product.RAINFOREST_RESIN]  # 直接取订单簿

        resin_take_orders, buy_order_volume, sell_order_volume = trader.take_orders(
            Product.RAINFOREST_RESIN,
            state.order_depths[Product.RAINFOREST_RESIN],
            trader.params[Product.RAINFOREST_RESIN]["fair_value"],
            trader.params[Product.RAINFOREST_RESIN]["take_width"],
            resin_position,
        )

        # 平仓
        resin_clear_orders, buy_order_volume, sell_order_volume = trader.clear_orders(
            Product.RAINFOREST_RESIN,
            state.order_depths[Product.RAINFOREST_RESIN],
            trader.params[Product.RAINFOREST_RESIN]["fair_value"],
            trader.params[Product.RAINFOREST_RESIN]["clear_width"],
            resin_position,
            buy_order_volume,
            sell_order_volume,
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