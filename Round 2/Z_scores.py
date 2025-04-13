from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Any
import json
import jsonpickle
import numpy as np



# 产品总览
class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET2 = "PICNIC_BASKET2"


# 产品参数总览
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
    Product.PICNIC_BASKET2: {"MAX_POS": 100}
}

BASKET_WEIGHTS = {Product.CROISSANTS: 6, Product.JAMS: 3, Product.DJEMBES: 1}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
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
        self.spread_mean = 48.76  # 初始均值
        self.spread_var = 85.12 ** 2  # 初始方差
        self.halflife = 695  # 半衰期
        self.lambda_ = 1 - np.exp(np.log(0.5) / self.halflife)  # 衰减因子
        self.components = {
            "CROISSANTS" : 6,
            "JAMS" : 3,
            "DJEMBES" : 1
        }

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
        window = prices[-self.WINDOW_SIZE :]  # 取最近的WINDOW_SIZE个价格
        return window[-1] - window[0]  # 终点减起点得到趋势

    def get_kelp_mean_price(self, product: str):
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
                orders = self.MRTraderObj.handle_resin_trading(state)

            if product == "PICNIC_BASKET1":
                orders = self.handle_basket_arbitrage1(state)

            if product == "KELP":
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
                            #print(
                                #f"KELP SELL {sell_volume} @ {best_bid} (Mean: {self.KELP_MEAN:.2f})"
                           # )
            if product == 'SQUID_INK' :
                # ink产品逻辑
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
            result[product] = orders
        traderData = json.dumps(
            {"price_history": self.price_history, "kelp_mean": self.KELP_MEAN}
        )

        return result, 0, traderData








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