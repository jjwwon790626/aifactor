import pandas as pd
import numpy as np

def depth_to_depth(signal, target_depth, origin_depth):
    
    raw_index = origin_depth['ts'].searchsorted(target_depth['ts']) - 1
    ob_index = pd.Series(raw_index)
    ob_index.loc[ob_index < 0] = 0
    target_feature = signal.loc[ob_index]
    target_feature.index = pd.RangeIndex(len(target_feature))
    return target_feature

def purged_train_test_split(X, y, test_size=None, purge_size=None):
    
    if test_size is None:
        test_size = 0.25
        
    if purge_size is None:
        purge_size = 0.1
    
    train_size = 1 - test_size - purge_size
    if not 0 < train_size < 1:
        raise AttributeError("test_size + purge_size shoule in (0, 1)")
            
    if not len(X) == len(y):
        raise AttributeError(f"X length {len(X)} and y length{len(y)} not equal")
    
    length = len(X)
    
    train_end = int(length * train_size)
    purge_end = int(length * purge_size) + train_end
    
    X_train, X_test = X[:train_end], X[purge_end:]
    y_train, y_test = y[:train_end], y[purge_end:]
    
    return X_train, X_test, y_train, y_test 

def train_test_split(X, y, test_size=None):
    return purged_train_test_split(X, y, test_size=test_size, purge_size=0)

def purged_single_split(X, test_size=None, purge_size=None):

    if test_size is None:
        test_size = 0.25
        
    if purge_size is None:
        purge_size = 0.1
    
    train_size = 1 - test_size - purge_size
    if not 0 < train_size < 1:
        raise AttributeError("test_size + purge_size shoule in (0, 1)")

    length = len(X)
    
    train_end = int(length * train_size)
    purge_end = int(length * purge_size) + train_end
    
    X_train, X_test = X[:train_end], X[purge_end:]
    
    return X_train, X_test

def single_split(X, test_size=None):
    return purged_single_split(X, test_size=test_size, purge_size=0)