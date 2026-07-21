import pandas as pd
import numpy as np

def depth_to_depth(signal, target_depth, origin_depth):
    
    raw_index = origin_depth['ts'].searchsorted(target_depth['ts']) - 1
    ob_index = pd.Series(raw_index)
    ob_index.loc[ob_index < 0] = 0
    target_feature = signal.loc[ob_index]
    target_feature.index = pd.RangeIndex(len(target_feature))
    return target_feature