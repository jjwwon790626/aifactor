from hft.utils.wrapper import trade_to_depth
import numpy as np
import pandas as pd

@trade_to_depth
def twap_diff(depth5, trade, n=20):
    return ((trade.v.abs() * trade.p).rolling(n).sum() / trade.v.abs().rolling(n).sum() - trade.p).fillna(0)