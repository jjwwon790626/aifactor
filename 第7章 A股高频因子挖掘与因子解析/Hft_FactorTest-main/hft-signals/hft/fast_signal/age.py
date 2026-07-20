import numpy as np
import pandas as pd
import numba as nb

@nb.jit
def get_age(x):
    last_value = x[-1]
    age = 0
    for i in range(2, len(x)):
        if x[-i] != last_value:
            break
        age += 1
    return age

def bid_age(depth5, trade, n=100):
        bp1 = depth5['bp1']
        bp1_changes = bp1.rolling(n).apply(get_age, engine='numba', raw=True).fillna(0)
        return bp1_changes

def bid_age(depth5, trade, n=100):
        ap1 = depth5['ap1']
        ap1_changes = ap1.rolling(n).apply(get_age, engine='numba', raw=True).fillna(0)
        return ap1_changes