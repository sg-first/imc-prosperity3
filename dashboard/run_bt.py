import pandas as pd
from datamodel import Listing, ConversionObservation
from round3_v3_merged import Trader
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
        symbol="PICNIC_BASKET2", product="PICNIC_BASKET2", denomination="PICNIC_BASKET2"),
    "VOLCANIC_ROCK": Listing(symbol="VOLCANIC_ROCK", product="VOLCANIC_ROCK", denomination="VOLCANIC_ROCK"),
    "VOLCANIC_ROCK_VOUCHER_9500": Listing(symbol="VOLCANIC_ROCK_VOUCHER_9500", product="VOLCANIC_ROCK_VOUCHER_9500", denomination="VOLCANIC_ROCK_VOUCHER_9500"),
    "VOLCANIC_ROCK_VOUCHER_9750": Listing(symbol=" VOLCANIC_ROCK_VOUCHER_9750", product="VOLCANIC_ROCK_VOUCHER_9750", denomination="VOLCANIC_ROCK_VOUCHER_9750"),
    "VOLCANIC_ROCK_VOUCHER_10000": Listing(symbol="VOLCANIC_ROCK_VOUCHER_10000", product="VOLCANIC_ROCK_VOUCHER_10000", denomination="VOLCANIC_ROCK_VOUCHER_10000"),
    "VOLCANIC_ROCK_VOUCHER_10250": Listing(symbol="VOLCANIC_ROCK_VOUCHER_10250", product="VOLCANIC_ROCK_VOUCHER_10250", denomination="VOLCANIC_ROCK_VOUCHER_10250"),
    "VOLCANIC_ROCK_VOUCHER_10500": Listing(symbol="VOLCANIC_ROCK_VOUCHER_10500", product="VOLCANIC_ROCK_VOUCHER_10500", denomination="VOLCANIC_ROCK_VOUCHER_10500"),
    "MAGNIFICENT_MACARONS":Listing(symbol="MAGNIFICENT_MACARONS", product="MAGNIFICENT_MACARONS", denomination="MAGNIFICENT_MACARONS")
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
    "VOLCANIC_ROCK":400,
    "VOLCANIC_ROCK_VOUCHER_9500":200,
    "VOLCANIC_ROCK_VOUCHER_9750":200,
    "VOLCANIC_ROCK_VOUCHER_10000":200,
    "VOLCANIC_ROCK_VOUCHER_10250":200,
    "VOLCANIC_ROCK_VOUCHER_10500":200,
    "MAGNIFICENT_MACARONS" : 75
}

# 这里写错了，函数需要实际定义，需要老李改一下
fair_calculations = {
    """
    "RAINFOREST_RESIN": calculate_rainforestResin_fair,
    "SQUID_INK": calculate_squidink_fair
    "KELP": calculate_kelp_fair
"""
}

day = 4
market_data = pd.read_csv(
    f"round-5-island-data-bottle/prices_round_5_day_{day}.csv", sep=";", header=0
)
trade_history = pd.read_csv(
    f"round-5-island-data-bottle/trades_round_5_day_{day}.csv", sep=";", header=0
)

observationsData = pd.read_csv(
    f"round-5-island-data-bottle/observations_round_5_day_{day}.csv", header=0
)

trader = Trader()
backtester = Backtester(
    trader,
    listings,
    position_limit,
    {},
    market_data,
    trade_history,
    observationsData,
    "trade_history_sim.log",

)  # 因为fair_calculations函数没写所以传空，backtester里找不到函数，会用中间价作为fair
backtester.run()
print(backtester.pnl)
