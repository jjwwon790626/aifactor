import numba
import pandas as pd
import numpy as np

def all_avg_return(ob, ask_sig, bid_sig, threshold=0.05, horizon_ticks=500):

    def _small_and_large_value(signal, threshold):

        threshold_length = int(len(signal) * threshold)
        small = np.argpartition(signal, threshold_length)[:threshold_length]
        big = np.argpartition(signal, -threshold_length)[-threshold_length:]

        small_value_threshold = max(signal[small])
        large_value_threshold = min(signal[big])

        return small_value_threshold, large_value_threshold

    ap1 = ob['ap1'].values
    bp1 = ob['bp1'].values
    price_mean = (ap1.mean() + bp1.mean())/2    
    ask_sig = pd.Series(ask_sig)
    bid_sig = pd.Series(bid_sig)

    ask_sig_small_threshold, ask_sig_large_threshold = _small_and_large_value(ask_sig, threshold)
    bid_sig_small_threshold, bid_sig_large_threshold = _small_and_large_value(bid_sig, threshold)
    
    ask_order_place = ask_sig < ask_sig_small_threshold
    bid_order_place = bid_sig > bid_sig_large_threshold

    ask_return, bid_return = all_return(ob, horizon_ticks)
    
    random_ask_filled_avg_return = -ask_return.mean() / price_mean
    random_bid_filled_avg_return = bid_return.mean() / price_mean

    signal_ask_avg_return = -ask_return[ask_order_place.index].mean() / price_mean
    signal_bid_avg_return = bid_return[bid_order_place.index].mean() / price_mean

    return random_ask_filled_avg_return, random_bid_filled_avg_return, signal_ask_avg_return, signal_bid_avg_return

def filled_avg_return(ob, ask_sig, bid_sig, threshold=0.05):
        
    def _small_and_large_value(signal, threshold):

        threshold_length = int(len(signal) * threshold)
        small = np.argpartition(signal, threshold_length)[:threshold_length]
        big = np.argpartition(signal, -threshold_length)[-threshold_length:]

        small_value_threshold = max(signal[small])
        large_value_threshold = min(signal[big])

        return small_value_threshold, large_value_threshold
    
    ap1 = ob['ap1'].values
    bp1 = ob['bp1'].values
    price_mean = (ap1.mean() + bp1.mean())/2    
    ask_sig = pd.Series(ask_sig)
    bid_sig = pd.Series(bid_sig)
    
    ask_sig_small_threshold, ask_sig_large_threshold = _small_and_large_value(ask_sig, threshold)
    bid_sig_small_threshold, bid_sig_large_threshold = _small_and_large_value(bid_sig, threshold)

    ask_return, bid_return = jit_purevalue(ap1, bp1)
    
    ask_return = pd.Series(ask_return)
    bid_return = pd.Series(bid_return)
    
    ask_filled = ask_return[ask_return!=0]
    bid_filled = bid_return[bid_return!=0]
    
    random_ask_filled_avg_return = -ask_filled.mean() / price_mean
    random_bid_filled_avg_return = bid_filled.mean() / price_mean
    
    ask_order_place = ask_sig < ask_sig_small_threshold
    bid_order_place = bid_sig > bid_sig_large_threshold
    
    signal_ask_filled_avg_return = -ask_return[ask_order_place.index].mean() / price_mean
    signal_bid_filled_avg_return = bid_return[bid_order_place.index].mean() / price_mean
    
    return random_ask_filled_avg_return, random_bid_filled_avg_return, signal_ask_filled_avg_return, signal_bid_filled_avg_return
    
def filled_return(ob, max_wait_ticks=400, filled_horizon_ticks=500, distance=0):
    return jit_purevalue(ob['ap1'].values, ob['bp1'].values, max_wait_ticks, filled_horizon_ticks, distance)
    
def all_return(ob, horizon_ticks):

    ap1 = ob.ap1.values
    bp1 = ob.bp1.values

    ap1_diff = np.roll(ap1, -horizon_ticks) - ap1
    bp1_diff = np.roll(bp1, -horizon_ticks) - bp1
    ap1_diff[-horizon_ticks:] = 0
    bp1_diff[-horizon_ticks:] = 0
    return ap1_diff, bp1_diff # equals to ask_return, bid_return

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
            plt.plot(mean_feature_values, (each_partition_corr), 'o-')
            plt.axhline(0, ls='--', c='black')
            plt.axvline(0, ls='--', c='black')

        else:
            fig.plot(mean_feature_values, (each_partition_corr), 'o-')
            fig.axhline(0, ls='--', c='black')
            fig.axvline(0, ls='--', c='black')
            
            
    return mean_feature_values, each_partition_corr

@numba.njit()
def jit_purevalue(ap1, bp1, max_wait_ticks=400, filled_horizon_ticks=500, distance=0):
    
    assert len(ap1) == len(bp1)
    length = len(ap1)
    mid_price = (ap1 + bp1) / 2

    bid_return = np.zeros(length)
    ask_return = np.zeros(length)
    
    bid_avg_fill_tick = np.zeros(length) - 1
    ask_avg_fill_tick = np.zeros(length) - 1
    
    for idx in range(length):
        
        # 末尾的值 直接写成0
        if idx + max_wait_ticks + filled_horizon_ticks >= length:
            bid_return[idx] = 0
            ask_return[idx] = 0
            continue
            
        ask_price = ap1[idx] + distance
        bid_price = bp1[idx] - distance
        
        for jdx in range(1, max_wait_ticks+1):
            if bp1[jdx+idx] > ask_price:
                ap_diff = ap1[jdx+idx+filled_horizon_ticks] - ask_price
                ask_return[idx] = ap_diff
                break
                
            # if jdx == max_wait_ticks:
            #    bid_return[idx]
        for jdx in range(1, max_wait_ticks+1):
            if ap1[jdx+idx] < bid_price:
                bp_diff = bp1[jdx+idx+filled_horizon_ticks] - bid_price
                bid_return[idx] = bp_diff
                break
                
    return ask_return, bid_return