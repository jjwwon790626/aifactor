from hft.utils.wrapper import trade_to_depth
import numpy as np
import pandas as pd

def weighted_price_to_mid(depth, trade, levels=5, alpha=1):
    avs = depth[get_columns("av", levels)]
    bvs = depth[get_columns("bv", levels)]
    aps = depth[get_columns("ap", levels)]
    bps = depth[get_columns("bp", levels)]
    if 0 < alpha < 1:
        decay_weights = np.array([alpha**n for n in range(levels)])
        avs *= decay_weights
        bvs *= decay_weights
    mp = (depth['ap1'] + depth['bp1'])/2
    return (avs.values * aps.values + bvs.values * bps.values).sum(axis=1) / (avs.values + bvs.values).sum(axis=1) - mp
    