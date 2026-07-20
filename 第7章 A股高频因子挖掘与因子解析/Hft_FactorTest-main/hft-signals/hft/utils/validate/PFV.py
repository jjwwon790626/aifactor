import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hft.utils.target import filled_return


# 评估高频交易中特征的做市收益潜力评估在不同特征值下进行买卖双边报价的预期收益


# def PFV_ask(feature, ob, partitions=100, price_lag=100):
# 
#     ask_return, _ = filled_return(ob, price_lag)
#     mid_price = (ob['ap1'] + ob['bp1'])/2
#     spread = (ob['ap1'] - ob['bp1']).mean()
#     length_feature = len(feature)
#     res = []
#     interval_floor = int(length_feature / partitions)
# 
#     sorted_feature = feature.sort_values()
# 
# 
#     for i in range(partitions):
#        if i == partitions - 1:
#            res.append(sorted_feature[i * interval_floor:])
#        else:
#            res.append(sorted_feature[i * interval_floor: (i+1) * interval_floor])
# 
#     mean_feature_values = np.zeros(len(res))
#     each_partition_corr = np.zeros((len(res),2))
# 
#     for i in range(len(res)):
#        idx = res[i].index
#        mean_feature_values[i] = sorted_feature.loc[idx].mean()
#       now_return_for_ask = (- ask_return.loc[idx].mean() + 0.5 * spread) / mean_mid_price * 10000

def PFV(ask_feature, bid_feature, ob, partitions=100, price_lag=500, maker_fee=-0.3, plot=False):

    ask_feature_index, ask_feature_return = feature_partition_freturn(ask_feature, ob, partitions, price_lag)
    bid_feature_index, bid_feature_return = feature_partition_freturn(bid_feature, ob, partitions, price_lag)

    ask_feature_return = ask_feature_return.T[0]
    bid_feature_return = bid_feature_return.T[1]

    ask_feature_return += abs(maker_fee)
    bid_feature_return += abs(maker_fee)
    
    plt.figure(figsize=[20,10])
    plt.plot(ask_feature_index, ask_feature_return, "r.-")
    plt.plot(bid_feature_index, bid_feature_return, "g.-")
    plt.axhline(ask_feature_return.mean(), c="r", ls='--')
    plt.axhline(bid_feature_return.mean(), c="g", ls='--')
    plt.axhline(0, c='black', ls='--', lw=5)
    plt.show()
    #return ask_feature_index, bid_feature_index, ask_feature_return, bid_feature_return
    


def feature_partition_freturn(feature, ob, partitions=100, price_lag=100, plot=False, fig=None):
    
    length_sort = len(feature)
    interval_floor = int(length_sort / partitions)
    res = []

    sorted_feature = feature.sort_values()
    
    # Assumed only use mid price lag as corr judgement method.
    mid_price = (ob['ap1'] + ob['bp1'])/2
    spread = (ob['ap1'] - ob['bp1']).mean()
    mean_mid_price = mid_price.mean()
    ask_return, bid_return = filled_return(ob, price_lag)
    ask_return = pd.Series(ask_return)
    bid_return = pd.Series(bid_return)
    

    for i in range(partitions):
        if i == partitions - 1:
            res.append(sorted_feature[i * interval_floor:])
        else:
            res.append(sorted_feature[i * interval_floor: (i+1) * interval_floor])

    mean_feature_values = np.zeros(len(res))
    each_partition_corr = np.zeros((len(res),2))

    for i in range(len(res)):
        idx = res[i].index
        mean_feature_values[i] = sorted_feature.loc[idx].mean()
        now_return_for_ask = (- ask_return.loc[idx].mean() + 0.5 * spread) / mean_mid_price * 10000
        now_return_for_bid = (bid_return.loc[idx].mean() + 0.5 * spread) / mean_mid_price * 10000
        
        if not now_return_for_ask:
            now_return_for_ask = 0
        if not now_return_for_bid:
            now_return_for_bid = 0

        each_partition_corr[i][0] = now_return_for_ask
        each_partition_corr[i][1] = now_return_for_bid
            
    if plot:

        if fig is None:
            plt.figure(figsize=[20,10])
            plt.plot(mean_feature_values, (each_partition_corr), 'ro-')
            plt.axhline(0, ls='--', c='black')
            plt.axvline(0, ls='--', c='black')

        else:
            fig.plot(mean_feature_values, (each_partition_corr), 'go-')
            fig.axhline(0, ls='--', c='black')
            fig.axvline(0, ls='--', c='black')
            
            
    return mean_feature_values, each_partition_corr