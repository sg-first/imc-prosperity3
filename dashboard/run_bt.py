import pandas as pd
from datamodel import Listing, ConversionObservation
from MYZscore import Trader
from backtester import Backtester

listings = {
    "RAINFOREST_RESIN": Listing(
        symbol="RAINFOREST_RESIN", product="RAINFOREST_RESIN", denomination="SEASHELLS"
    ),
    "SQUID_INK": Listing(
        symbol="SQUID_INK", product="SQUID_INK", denomination="SEASHELLS"
    ),
    "KELP": Listing(symbol="KELP", product="KELP", denomination="SEASHELLS"),
    "DJEMBES": Listing(symbol="DJEMBES", product="DJEMBES", denomination="DJEMBES"),
    "CROISSANTS": Listing(
        symbol="CROISSANTS", product="CROISSANTS", denomination="CROISSANTS"
    ),
    "JAMS": Listing(symbol="JAMS", product="JAMS", denomination="JAMS"),
    "PICNIC_BASKET1": Listing(
        symbol="PICNIC_BASKET1", product="PICNIC_BASKET1", denomination="PICNIC_BASKET1"
    ),
    "PICNIC_BASKET2": Listing(
        symbol="PICNIC_BASKET2", product="PICNIC_BASKET2", denomination="PICNIC_BASKET2"
    ),
}

position_limit = {
    "RAINFOREST_RESIN": 50,
    "SQUID_INK": 50,
    "KELP": 50,
    "DJEMBES": 60,
    "CROISSANTS": 250,
    "JAMS": 350,
    "PICNIC_BASKET1": 60,
    "PICNIC_BASKET2": 100,
}

# 这里写错了，函数需要实际定义，需要老李改一下
fair_calculations = {
    """
    "RAINFOREST_RESIN": calculate_rainforestResin_fair,
    "SQUID_INK": calculate_squidink_fair
    "KELP": calculate_kelp_fair
"""
}

day = 1
market_data = pd.read_csv(
    f"../Round 2/round-2-island-data-bottle/prices_round_2_day_{day}.csv", sep=";", header=0
)
trade_history = pd.read_csv(
    f"../round-2-island-data-bottle/trades_round_2_day_{day}.csv", sep=";", header=0
)

trader = Trader()
backtester = Backtester(
    trader,
    listings,
    position_limit,
    {},
    market_data,
    trade_history,
    "trade_history_sim.log",
)  # 因为fair_calculations函数没写所以传空，backtester里找不到函数，会用中间价作为fair
backtester.run()
print(backtester.pnl)
