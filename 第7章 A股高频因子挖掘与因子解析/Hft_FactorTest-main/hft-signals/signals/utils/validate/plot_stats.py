
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator


from signals.validate.feature_partition_corr import feature_partition_corr
from signals.validate.others_corr import others_corr
from signals.validate.feature_hist import feature_hist
from signals.validate.return_corr import return_corr
from statsmodels.graphics.tsaplots import plot_pacf


CORR_INTERVAL_DEFAULT = np.arange(5, 200, 30).tolist() + np.arange(220, 2000, 50).tolist()

def feature_partition_corr(feature, ob, partitions=100, price_lag=50, plot=False, fig=None):
    
    length_sort = len(feature)
    interval_floor = int(length_sort / partitions)
    res = []

    sorted_feature = feature.sort_values()
    
    # Assumed only use mid price lag as corr judgement method.
    mid_price = (ob['ap1'] + ob['bp1'])/2
    changes = mid_price.diff(price_lag).shift(-price_lag).fillna(0)

    for i in range(partitions):
        if i == partitions - 1:
            res.append(sorted_feature[i * interval_floor:])
        else:
            res.append(sorted_feature[i * interval_floor: (i+1) * interval_floor])

    mean_feature_values = []
    each_partition_corr = []

    for i in range(len(res)):
        idx = res[i].index
        mean_feature_values.append(sorted_feature.loc[idx].mean())
        now_corr = np.corrcoef(sorted_feature.loc[idx], changes.loc[idx])[0][1]
        if np.isnan(now_corr):
            now_corr = 0.00001
        each_partition_corr.append(now_corr)

    if plot:

        if fig is None:
            plt.figure(figsize=[20,10])
            plt.plot(mean_feature_values, np.cumsum(each_partition_corr), 'o-')
            plt.axhline(0, ls='--', c='black')
            plt.axvline(0, ls='--', c='black')

        else:
            fig.plot(mean_feature_values, np.cumsum(each_partition_corr), 'o-')
            fig.axhline(0, ls='--', c='black')
            fig.axvline(0, ls='--', c='black')
            
            
    return {"mean_feature_values": mean_feature_values, "each_partition_corr": each_partition_corr}


def feature_partition_return(feature, ob, partitions=100, price_lag=500, plot=False, fig=None):
    
    length_sort = len(feature)
    interval_floor = int(length_sort / partitions)
    res = []

    sorted_feature = feature.sort_values()
    
    # Assumed only use mid price lag as corr judgement method.
    mid_price = (ob['ap1'] + ob['bp1'])/2
    changes = mid_price.diff(price_lag).shift(-price_lag).fillna(0)

    for i in range(partitions):
        if i == partitions - 1:
            res.append(sorted_feature[i * interval_floor:])
        else:
            res.append(sorted_feature[i * interval_floor: (i+1) * interval_floor])

    mean_feature_values = []
    each_partition_corr = []

    for i in range(len(res)):
        idx = res[i].index
        mean_feature_values.append(sorted_feature.loc[idx].mean())
        now_corr = changes.loc[idx].mean()
        if np.isnan(now_corr):
            now_corr = 0.00001
        each_partition_corr.append(now_corr)

    if plot:

        if fig is None:
            plt.figure(figsize=[20,10])
            plt.plot(mean_feature_values, (each_partition_corr), 'o-')
            plt.axhline(0, ls='--', c='black')
            plt.axvline(0, ls='--', c='black')

        else:
            fig.plot(mean_feature_values, (each_partition_corr), 'o-')
            fig.axhline(0, ls='--', c='black')
            fig.axvline(0, ls='--', c='black')
            
            
    return {"mean_feature_values": mean_feature_values, "each_partition_corr": each_partition_corr}


def others_corr(feature, others, method="pearson", plot=False, fig=None):

    def others_corr_plot(res, xlabel, fig=None):
        values = dict(zip(xlabel, res))
        sorted_dict = dict(sorted(values.items(), key=operator.itemgetter(1)))
        if fig == None:
            fig = plt
            plt.figure(figsize=[10,12])
        fig.axvline(0.7, ls='--', c='black')
        ys = [abs(x) for x in sorted_dict.values()]
        bars = fig.barh(list(sorted_dict.keys()),ys)
        fig.bar_label(bars, list(sorted_dict.keys()), padding=5)

    res = np.corrcoef(feature.values, [others[name].values for name in others.columns])[0][1:]
    if plot:
        xlabel = others.columns
        others_corr_plot(res, xlabel, fig)
    return res


def feature_hist(feature, range_threshold=0.002, bars_amount=100, plot=False, fig=None):

    def hist_plot(data, arange, bars_amount, fig):

        if fig is None:
            plt.hist(data, bars_amount, range=arange)
        else:
            fig.hist(data, bars_amount, range=arange)

    feature_length = len(feature)
    small_idx = int(feature_length * range_threshold)
    large_idx = int(feature_length * (1 - range_threshold))
    sorted_feature = feature.sort_values()
    small_value = sorted_feature.iloc[small_idx]
    large_value = sorted_feature.iloc[large_idx]
    arange = [small_value, large_value]
    if plot:
        hist_plot(feature, arange, bars_amount, fig)


def partial_corr_hist(feature, orderbook, plot=False, fig=None):
        
    def partial_corr(ob, feature):
        corr = []
        mp = (ob.ap1 + ob.bp1)/2
        temp_lag = mp.diff(0)
        for idx, lag in enumerate(CORR_INTERVAL_DEFAULT):
            now_temp = mp.diff(lag).shift(-lag).fillna(0)
            corr.append(np.corrcoef(now_temp - temp_lag, feature)[0][1])
            temp_lag = now_temp
        return corr

    def partial_plot(x, y, fig):
        fig.bar(x,y,width=50)

    x = CORR_INTERVAL_DEFAULT
    y = partial_corr(orderbook, feature)
    if plot:
        partial_plot(x, y, fig)


