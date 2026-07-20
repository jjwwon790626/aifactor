import pandas as pd
import numpy as np

def trade_to_depth(trade_signal):
    def wrapper(*args, **kwargs):
        original_trade_signal = trade_signal(*args, **kwargs)

        if kwargs.get("datas"):
            ob = kwargs['datas']['depth5']
            tr = kwargs['datas']['trade']
        else:
            ob = kwargs['depth5']
            tr = kwargs['trade']

        raw_index = tr['ts'].searchsorted(ob['ts']) - 1
        ob_index = pd.Series(raw_index)
        ob_index.loc[ob_index < 0] = 0

        orderbook_trade_feature = original_trade_signal.loc[ob_index]
        orderbook_trade_feature.index = pd.RangeIndex(len(orderbook_trade_feature))
        
        return orderbook_trade_feature
    return wrapper
    