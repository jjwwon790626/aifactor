import numba as nb
import numpy as np
import pandas as pd


def ask_label(orderbook=None, trade=None, long_ratio=1.0, short_ratio=0.1, vertical_span=300, volatility_estimate_span=1000, nofilled_replace=True):

    L = Labelist(orderbook, long_ratio, short_ratio, vertical_span, volatility_estimate_span, nofilled_replace=nofilled_replace)
    return L.get_ask_signal()

def bid_label(orderbook=None, trade=None, long_ratio=1.0, short_ratio=0.1, vertical_span=300, volatility_estimate_span=1000, nofilled_replace=True):
    
    L = Labelist(orderbook, long_ratio, short_ratio, vertical_span, volatility_estimate_span, nofilled_replace=nofilled_replace)
    return L.get_bid_signal()

def const_ask_label(orderbook=None, trade=None, win_basepoints=0.5, loss_basepoints=3, vertical_span=300, nofilled_replace=True):

    L = ConstLabelist(orderbook, win_basepoints, loss_basepoints, vertical_span, nofilled_replace=nofilled_replace)
    return L.get_ask_signal()

def const_bid_label(orderbook=None, trade=None, win_basepoints=0.5, loss_basepoints=3, vertical_span=300, nofilled_replace=True):

    L = ConstLabelist(orderbook, win_basepoints, loss_basepoints, vertical_span, nofilled_replace=nofilled_replace)
    return L.get_bid_signal()


class Labelist:
    
    def __init__(self, orderbook, long_ratio, short_ratio, vertical_span, volatility_estimate_span=1000, nofilled_replace=True):
        self.orderbook = orderbook.copy()
        self.long_ratio = long_ratio
        self.short_ratio = short_ratio
        self.vertical_span = vertical_span
        self.volatility_estimate_span = volatility_estimate_span
        self.nofilled_replace = nofilled_replace
        
        self._preprocess()
        
    def _preprocess(self):
        
        mid_price = (self.orderbook['ap1'] + self.orderbook['bp1'])/2
        self.orderbook['mid_price'] = mid_price.fillna(0)
        self.orderbook['volmp'] = self.calculate_volatility()
        self.orderbook['index'] = self.orderbook.index

        self._set_barrier()

    def _set_barrier(self):
        self.orderbook['long_barrier'] = self.orderbook['volmp'] * self.long_ratio
        self.orderbook['short_barrier'] = self.orderbook['volmp'] * self.short_ratio
        
    def calculate_volatility(self, method="ewma"):
        return self.orderbook['mid_price'].ewm(span=self.volatility_estimate_span).std().fillna(0)
    
    def not_filled(self, span=200):
        is_now_ask_min = self.orderbook[::-1].ap1.rolling(span).min() == self.orderbook[::-1].ap1
        is_now_bid_max = self.orderbook[::-1].bp1.rolling(span).max() == self.orderbook[::-1].bp1
        
        return is_now_ask_min, is_now_bid_max
    
    def get_not_filled(self):

        vertical_span = self.vertical_span
        span_max_ask = self.orderbook['ap1'][::-1].rolling(vertical_span).max()
        span_min_bid = self.orderbook['bp1'][::-1].rolling(vertical_span).min()
        is_now_ask_max = self.orderbook['ap1'] == span_max_ask.sort_index()
        is_now_bid_min = self.orderbook['bp1'] == span_min_bid.sort_index()
        return is_now_ask_max, is_now_bid_min
        
    def get_bid_signal(self):

        if self.nofilled_replace:
            _, bid_min = self.get_not_filled()

        bid_prices = np.array(self.orderbook['bp1'])
        stop_loss_return = -np.array(self.orderbook['long_barrier'])
        stop_profit_return = np.array(self.orderbook['short_barrier'])
        
        profit_hit, loss_hit = self.get_signal_loop(bid_prices, stop_profit_return, stop_loss_return, self.vertical_span)
        
        bid_win_position = pd.Series(profit_hit > loss_hit)
        bid_loss_position = pd.Series(profit_hit < loss_hit)
        bid_hold_position = pd.Series(profit_hit == loss_hit)
        
        bid_signal = pd.DataFrame(columns=['bid_label'], index=bid_win_position.index)

        bid_signal.loc[bid_win_position, 'bid_label'] = 1
        bid_signal.loc[bid_loss_position, 'bid_label'] = -1
        bid_signal.loc[bid_hold_position, 'bid_label'] = 0
        
        if self.nofilled_replace:
            bid_signal.loc[bid_min, 'bid_label'] = 0
        
        return bid_signal
        
    def get_ask_signal(self):
        
        if self.nofilled_replace:
            ask_max, _ = self.get_not_filled()

        ask_prices = np.array(self.orderbook['ap1'])
        stop_loss_return = np.array(self.orderbook['long_barrier'])
        stop_profit_return = -np.array(self.orderbook['short_barrier'])
        
        loss_hit, profit_hit = self.get_signal_loop(ask_prices, stop_loss_return, stop_profit_return, self.vertical_span)
        
        ask_win_position = pd.Series(profit_hit > loss_hit)
        ask_loss_position = pd.Series(profit_hit < loss_hit)
        ask_hold_position = pd.Series(profit_hit == loss_hit)
        
        ask_signal = pd.DataFrame(columns=['ask_label'], index=ask_win_position.index)
        
        ask_signal.loc[ask_win_position, 'ask_label'] = 1
        ask_signal.loc[ask_loss_position, 'ask_label'] = -1
        ask_signal.loc[ask_hold_position, 'ask_label'] = 0
        
        if self.nofilled_replace:
            ask_signal.loc[ask_max, 'ask_label'] = 0
        
        return ask_signal
        
    @staticmethod
    @nb.jit(nopython=True)
    def get_signal_loop(prices, ub, lb, vertical_span=200):
        ubs = [0] * len(prices)
        lbs = [0] * len(prices)
        length = len(prices)
        for i in range(length):
            end = min(i+vertical_span, length-1)
            for j in range(1+i, end):
                if prices[j] > prices[i] + ub[i]:
                    ubs[i] = float(j)
                    break
            for j in range(1+i, end):
                if prices[j] < prices[i] + lb[i]:
                    lbs[i] = float(j)
                    break
            if len(ubs) < i:
                ubs[i] = float(end)
            if len(lbs) < i:
                lbs[i] = float(end)
        return np.array(ubs), np.array(lbs)
    
class ConstLabelist(Labelist):

    def __init__(self, orderbook, win_basepoints, loss_basepoints, vertical_span, nofilled_replace=True):
        self.orderbook = orderbook.copy()
        self.long_ratio = win_basepoints
        self.short_ratio = loss_basepoints
        self.vertical_span = vertical_span
        self.nofilled_replace = nofilled_replace
        
        self._preprocess()

    def _preprocess(self):
        
        mid_price = (self.orderbook['ap1'] + self.orderbook['bp1'])/2
        self.orderbook['mid_price'] = mid_price.fillna(0)
        self.orderbook['index'] = self.orderbook.index

        self._set_barrier()
        
    def _set_barrier(self):
        self.orderbook['long_barrier'] = self.orderbook['bp1'] / 10000 * self.long_ratio
        self.orderbook['short_barrier'] = self.orderbook['ap1'] / 10000 * self.short_ratio