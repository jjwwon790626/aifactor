from .IFactor import IFactor
import numpy as np
import numba as nb
import pandas as pd
from hft.utils.wrapper import trade_to_depth


@nb.jit(nopython=True)
def get_volume_at_same_price(prices, volumes, n):
    length = len(prices)
    res = np.zeros(length)
    
    for i in range(n, length):
        temp = 0
        last_price = prices[i]
        for j in range(1, n):
            if prices[i-j] != last_price:
                break
            temp += volumes[i-j]
        res[i] = temp
    return res
        

class volume_at_same_price(IFactor):

    describe = {
        "name": "volume_at_same_price",
        "datas": [
            "trade"
        ],
        "params": [
            {
                "name": "n",
                "default_value": "100"
            }
        ],
        "description": """
        volume_at_same_price.
        """
    }

    @trade_to_depth
    def main(cls, **kwargs):
        trade = kwargs.get("datas", {}).get("trade")
        n = kwargs.get("params", {}).get("n")

        prices = trade.p.values
        volumes = trade.v.values

        return pd.Series(get_volume_at_same_price(prices, volumes, n))

