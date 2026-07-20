import numpy as np
import pandas as pd

def oir_diff(depth5, trade, length=10):

    a_volumes = depth5.av1 + depth5.av2 + depth5.av3 + depth5.av4 + depth5.av5
    b_volumes =  -(- depth5.bv1 - depth5.bv2 - depth5.bv3 - depth5.bv4 - depth5.bv5)
    oiroir = (np.log(b_volumes + 1) / ((np.log(b_volumes + 1) + np.log(a_volumes + 1))) - 0.5) * 2
    return pd.Series(oiroir.diff(10).fillna(0) - oiroir.diff().fillna(0))