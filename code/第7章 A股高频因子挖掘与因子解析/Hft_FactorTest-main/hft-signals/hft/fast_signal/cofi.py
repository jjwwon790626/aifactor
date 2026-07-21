import numpy as np
import pandas as pd

def cofi(depth5, trade):

    a = depth5['bv1']*np.where(depth5['bp1'].diff()>=0,1,0)
    b = depth5['bv1'].shift()*np.where(depth5['bp1'].diff()<=0,1,0)
    c = depth5['av1']*np.where(depth5['ap1'].diff()<=0,1,0)
    d = depth5['av1'].shift()*np.where(depth5['ap1'].diff()>=0,1,0)

    return (a - b - c + d).fillna(0)