from hft.utils.wrapper import trade_to_depth
import numpy as np
import pandas as pd

@trade_to_depth
def volume(depth5, trade):
    return trade.v.cumsum()

def pmtv(depth5, trade):
    _pmtv = abs(volume(depth5=depth5, trade=trade).diff().fillna(0)) * depth5.bp1.diff().fillna(0)
    return _pmtv