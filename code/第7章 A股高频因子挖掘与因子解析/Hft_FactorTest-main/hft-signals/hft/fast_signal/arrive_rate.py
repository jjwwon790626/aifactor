from hft.utils.wrapper import trade_to_depth
import numpy as np
import pandas as pd

@trade_to_depth
def arrive_rate(depth5, trade, n=300):
    res = trade['ts'].diff(n).fillna(0) / n
    return res