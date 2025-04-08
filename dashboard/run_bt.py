import datetime
import json
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math
from datamodel import Listing, ConversionObservation
from baseline import Trader
from backtester import Backtester

listings = {
    'RAINFOREST_RESIN': Listing(symbol='RAINFOREST_RESIN', product='RAINFOREST_RESIN', denomination='SEASHELLS'),
    'SQUID_INK': Listing(symbol='SQUID_INK', product='SQUID_INK', denomination='SEASHELLS'),
    'KELP': Listing(symbol='KELP', product='KELP', denomination='SEASHELLS')
}

position_limit = {
    'RAINFOREST_RESIN': 50,
    'SQUID_INK': 50,
    'KELP': 50
}

# fix:这里写错了，函数需要实际定义，@老李 改一下
fair_calculations = {'''
    "RAINFOREST_RESIN": calculate_rainforestResin_fair,
    "SQUID_INK": calculate_squidink_fair
    "KELP": calculate_kelp_fair
'''}

day = 0
market_data = pd.read_csv(f"./round-1-island-data-bottle/prices_round_1_day_{day}.csv", sep=";", header=0)
trade_history = pd.read_csv(f"./round-1-island-data-bottle/trades_round_1_day_{day}.csv", sep=";", header=0)

trader = Trader()
backtester = Backtester(trader, listings, position_limit, fair_calculations, market_data, trade_history, "trade_history_sim.log")
backtester.run()
print(backtester.pnl)