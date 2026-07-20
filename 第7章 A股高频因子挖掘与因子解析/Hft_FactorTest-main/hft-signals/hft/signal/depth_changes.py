from .IFactor import IFactor
import numpy as np
import numba as nb

@nb.jit
def get_change(x):
    age = 0
    for i in range(2, len(x)):
        if x[-1] != x[-i-1]:
            age += 1
    return age

class depth_bid_change(IFactor):

    describe = {
        "name": "depth_bid_change",
        "datas": [
            "depth5"
        ],
        "params": [{
            "name": "n",
            "default_value": 100,
        }],
        "description": """
        Orderbook Change.
        """
    }
    
    def main(cls, **kwargs):
        depth5 = kwargs.get("datas", {}).get("depth5")
        n = kwargs.get("params", {}).get("n")

        if depth5 is None:
            depth5 = kwargs.get("orderbook")
        bp1 = depth5['bp1']
        bp1_changes = bp1.rolling(n).apply(get_change, engine='numba', raw=True).fillna(0)
        return bp1_changes

class depth_ask_change(IFactor):

    describe = {
        "name": "depth_ask_change",
        "datas": [
            "depth5"
        ],
        "params": [{
            "name": "n",
            "default_value": 20,
        }],
        "description": """
        Orderbook Change.
        """
    }
    
    def main(cls, **kwargs):
        depth5 = kwargs.get("datas", {}).get("depth5")
        n = kwargs.get("params", {}).get("n")

        if depth5 is None:
            depth5 = kwargs.get("orderbook")
        ap1 = depth5['ap1']
        ap1_changes = ap1.rolling(n).apply(get_change, engine='numba', raw=True).fillna(0)
        return ap1_changes