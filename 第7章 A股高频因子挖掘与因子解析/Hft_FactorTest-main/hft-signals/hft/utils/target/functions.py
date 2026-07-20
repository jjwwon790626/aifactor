from .label import const_ask_label, const_bid_label, ask_label, bid_label
from .mid_price_changes import filled_avg_return, all_avg_return,  all_return, filled_return

########################################################
################### Three Way Labels ###################
########################################################

def vol_label(kwargs=None):

    description = {
        "datas":{
            "depth5"
        },
        "side": "bid or ask",
        "params":{
            "long_ratio": float,
            "short_ratio": float,
            "waiting_window_length": int,
            "volatility_estimate_span": int
        }
    }

    if kwargs is None:
        print(description)
        return

    datas = kwargs.get("datas")
    if datas is None:
        raise AttributeError("{datas} not in inputs")
    depth5 = datas.get("depth5")
    trade = datas.get("trade")
    if depth5 is None:
        raise AttributeError("{depth5 or trade} not in {datas} input")
    
    side = kwargs.get("side")
    if side not in ['bid', 'ask']:
        raise AttributeError("{side} not correct option, should be ask/bid")

    params = kwargs.get("params", {})
    long_ratio = params.get("long_ratio") or 0.2
    short_ratio = params.get("short_ratio") or 0.2
    vertical_span = params.get("waiting_window_length") or 300
    volatility_estimate_span = params.get("volatility_estimate_span") or 1000
    nofilled_replace = True

    if side == 'bid':
        return bid_label(orderbook=depth5, trade=trade, long_ratio=long_ratio, short_ratio=short_ratio, vertical_span=vertical_span, volatility_estimate_span=volatility_estimate_span, nofilled_replace=nofilled_replace)
    elif side == 'ask':
        return ask_label(orderbook=depth5, trade=trade, long_ratio=long_ratio, short_ratio=short_ratio, vertical_span=vertical_span, volatility_estimate_span=volatility_estimate_span, nofilled_replace=nofilled_replace)
    
def const_label(kwargs=None):
    
    description = {
        "datas":{
            "depth5"
        },
        "side": "bid or ask",
        "params":{
            "win_basepoints": float,
            "loss_basepoints": float,
            "waiting_window_length": int,
        }
    }

    if kwargs is None:
        print(description)
        return

    datas = kwargs.get("datas")
    if datas is None:
        raise AttributeError("{datas} not in inputs")
    depth5 = datas.get("depth5")
    trade = datas.get("trade")
    if depth5 is None:
        raise AttributeError("{depth5 or trade} not in {datas} input")
    
    side = kwargs.get("side")
    if side not in ['bid', 'ask']:
        raise AttributeError("{side} not correct option, should be ask/bid")

    params = kwargs.get("params", {})
    win_basepoints = params.get("win_basepoints") or 1
    loss_basepoints = params.get("loss_basepoints") or 1
    vertical_span = params.get("waiting_window_length") or 300
    nofilled_replace = True

    if side == 'bid':
        return const_bid_label(orderbook=depth5, trade=trade, win_basepoints=win_basepoints, loss_basepoints=loss_basepoints, vertical_span=vertical_span, nofilled_replace=nofilled_replace)
    elif side == 'ask':
        return const_ask_label(orderbook=depth5, trade=trade, win_basepoints=win_basepoints, loss_basepoints=loss_basepoints, vertical_span=vertical_span, nofilled_replace=nofilled_replace)

########################################################
################### Mid Price Changes ##################
########################################################

def mp_changes(kwargs=None):

    description = {
        "datas":{
            "depth5"
        },
        "params":{
            "window_length": int,
        }
    }

    if kwargs is None:
        print(description)
        return

    datas = kwargs.get("datas")
    if datas is None:
        raise AttributeError("{datas} not in inputs")
    depth5 = datas.get("depth5")
    if depth5 is None:
        raise AttributeError("{depth5} not in {datas} input")
    
    params = kwargs.get("params", {})
    window_length = params.get("window_length")

    ask_return, bid_return = all_return(depth5, window_length)
    return {"ask_return": ask_return, "bid_return": bid_return}

def filled_mp_changes(kwargs=None):

    description = {
        "datas":{
            "depth5"
        },
        "params":{
            "max_wait_ticks": int,
            "filled_horizon_ticks": int
        }
    }

    if kwargs is None:
        print(description)
        return

    datas = kwargs.get("datas")
    if datas is None:
        raise AttributeError("{datas} not in inputs")
    depth5 = datas.get("depth5")
    if depth5 is None:
        raise AttributeError("{depth5} not in {datas} input")
    
    params = kwargs.get("params", {})
    window_length = params.get("window_length")

    ask_return, bid_return = all_return(depth5, window_length)
    return {"ask_return": ask_return, "bid_return": bid_return}
