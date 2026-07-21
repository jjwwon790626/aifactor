import numba as nb
import numpy as np
import pandas as pd

@nb.njit
def _ask_withdraws_volume(l, n, levels=5):
    withdraws = 0
    for price_index in range(2,2+4*levels, 4):
        now_p = n[price_index]
        for price_last_index in range(2,2+4*levels,4):
            if l[price_last_index] == now_p:
                withdraws -= min(n[price_index+1] - l[price_last_index + 1], 0)
        
    return withdraws

@nb.njit
def _bid_withdraws_volume(l, n):
    withdraws = 0
    for price_index in range(0,4*levels, 4):
        now_p = n[price_index]
        for price_last_index in range(0,4*levels,4):
            if l[price_last_index] == now_p:
                withdraws -= min(n[price_index+1] - l[price_last_index + 1], 0)
        
    return withdraws

def ask_withdraws(depth5, trade):
    ob_values = depth5.values
    flows = np.zeros(len(ob_values))
    for i in range(1, len(ob_values)):
        flows[i] = _ask_withdraws_volume(ob_values[i-1], ob_values[i])
    return pd.Series(flows)

def bid_withdraws(depth5, trade):
    ob_values = depth5.values
    flows = np.zeros(len(ob_values))
    for i in range(1, len(ob_values)):
        flows[i] = _bid_withdraws_volume(ob_values[i-1], ob_values[i])
    return pd.Series(flows)

def withdraws_diff(depth5, trade):
    return ask_withdraws(depth5, trade) - bid_withdraws(depth5, trade)