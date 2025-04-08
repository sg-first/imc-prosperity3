import math
import json
from typing import List
from datamodel import OrderDepth, Order, TradingState

class Trader:
    def __init__(self):
        self.price_history = {}
        self.MAX_POS = 20
        # 为不同产品设置振幅阈值。下面的商品名和对应的阈值要更改
        self.AMPLITUDE_THRESHOLDS = {
            "PEARLS": {
                "min": 2,    # PEARLS波动较小，设置较小的阈值
                "max": 10
            },
            "BANANAS": {
                "min": 5,    # BANANAS波动较大，设置较大的阈值
                "max": 20
            }
        }
        # 默认阈值（用于新产品）
        self.DEFAULT_AMPLITUDE = {
            "min": 3,
            "max": 15
        }

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        """计算当前中间价"""
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0
        if best_bid == 0 or best_ask == 0:
            return best_bid + best_ask
        return (best_bid + best_ask) / 2

    def wave_trading_with_amplitude(self, product: str, prices: list, current_price: float, 
                                  current_position: int, amplitude_window: int = 20,
                                  phase_buy_threshold: float = math.pi / 3,
                                  phase_sell_threshold: float = 5 * math.pi / 3,
                                  max_position: int = 20):
        """
        改进的波段策略，加入振幅阈值控制
        """
        if len(prices) < amplitude_window:
            return "HOLD", None, 0

        # 计算当前振幅
        window = prices[-amplitude_window:]
        highest = max(window)
        lowest = min(window)
        current_amplitude = highest - lowest

        # 获取该产品的振幅阈值
        amplitude_thresholds = self.AMPLITUDE_THRESHOLDS.get(
            product, self.DEFAULT_AMPLITUDE
        )
        min_amplitude = amplitude_thresholds["min"]
        max_amplitude = amplitude_thresholds["max"]

        # 振幅过滤
        if current_amplitude < min_amplitude or current_amplitude > max_amplitude:
            return "HOLD", None, current_amplitude

        # 计算相位
        if current_amplitude == 0:
            normalized_position = 0.5
        else:
            normalized_position = (current_price - lowest) / current_amplitude
        phase = normalized_position * 2 * math.pi

        # 生成交易信号
        if phase < phase_buy_threshold and current_position < max_position:
            signal = "BUY"
        elif phase > phase_sell_threshold and current_position > -max_position:
            signal = "SELL"
        else:
            signal = "HOLD"

        return signal, phase, current_amplitude

    def run(self, state: TradingState):
        result = {}
        
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders: List[Order] = []

            # 获取当前持仓
            current_pos = state.position.get(product, 0)

            # 计算并更新价格历史
            mid_price = self.get_mid_price(order_depth)
            if product not in self.price_history:
                self.price_history[product] = []
            self.price_history[product].append(mid_price)

            # 控制历史数据量
            amplitude_window = 20
            if len(self.price_history[product]) > amplitude_window:
                self.price_history[product].pop(0)

            # 当有足够的历史数据时执行策略
            if len(self.price_history[product]) >= amplitude_window:
                signal, phase, current_amplitude = self.wave_trading_with_amplitude(
                    product=product,
                    prices=self.price_history[product],
                    current_price=mid_price,
                    current_position=current_pos,
                    amplitude_window=amplitude_window,
                    phase_buy_threshold=math.pi / 3,
                    phase_sell_threshold=5 * math.pi / 3,
                    max_position=self.MAX_POS
                )

                # 获取最优价格
                best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
                best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0

                # 执行交易
                if signal == "BUY" and best_ask:
                    ask_volume = abs(order_depth.sell_orders[best_ask])
                    buy_volume = min(ask_volume, self.MAX_POS - current_pos)
                    if buy_volume > 0:
                        orders.append(Order(product, best_ask, buy_volume))
                        print(f"{product} BUY {buy_volume} @ {best_ask} Amplitude: {current_amplitude:.2f}")

                elif signal == "SELL" and best_bid:
                    bid_volume = abs(order_depth.buy_orders[best_bid])
                    sell_volume = min(bid_volume, self.MAX_POS + current_pos)
                    if sell_volume > 0:
                        orders.append(Order(product, best_bid, -sell_volume))
                        print(f"{product} SELL {sell_volume} @ {best_bid} Amplitude: {current_amplitude:.2f}")

            result[product] = orders

        # 记录状态信息
        traderData = json.dumps({
            "price_history": self.price_history,
            "positions": state.position
        })

        return result, 0, traderData