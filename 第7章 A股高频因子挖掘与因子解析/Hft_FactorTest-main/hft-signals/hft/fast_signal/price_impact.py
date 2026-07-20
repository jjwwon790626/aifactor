from hft.utils.wrapper import trade_to_depth
import numpy as np
import pandas as pd

def price_impact(depth5, trade, n=5):
    ask, bid, ask_v, bid_v = 0, 0, 0, 0
    for i in range(1, n+1):
        ask += depth5[f'ap{i}'] * depth5[f'av{i}']
        bid += depth5[f'bp{i}'] * depth5[f'bv{i}']
        ask_v += depth5[f'av{i}']
        bid_v += depth5[f'bv{i}']
    ask /= ask_v
    bid /= bid_v
    return pd.Series(-(depth5['ap1'] - ask)/depth5['ap1'] - (depth5['bp1'] - bid)/depth5['bp1'], name="price_impact")