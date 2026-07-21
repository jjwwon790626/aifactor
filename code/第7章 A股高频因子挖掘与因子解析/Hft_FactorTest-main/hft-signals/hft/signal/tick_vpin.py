from os import stat
from .IFactor import IFactor
from hft.utils.wrapper import trade_to_depth
import pandas as pd

@trade_to_depth
def turnover_agg(depth5=None, trade=None):
    return (trade.v.abs() * trade.p).cumsum()
    

@trade_to_depth
def volume_agg(depth5=None, trade=None):
    return trade.v.abs().cumsum()


class tick_vpin(IFactor):

    describe = {
        "name": "tick_vpin",
        "datas": [
            "depth5",
            "trade"
        ],
        "params": [{
            "name": "n",
            "default": 10
        }],
        "description": """
        Tick VPIN.
        """
    }
    
    def main(cls, **kwargs):
        depth5 = kwargs.get("datas").get("depth5")
        trade = kwargs.get("datas").get("trade")
        n = kwargs.get("params").get("n")

        bp1 = depth5.bp1
        ap1 = depth5.ap1
        turnover = turnover_agg(depth5=depth5, trade=trade).diff().fillna(0)
        volume = volume_agg(depth5=depth5, trade=trade).diff().fillna(0)
        
        avg_price = turnover/volume
        vdiff = (avg_price - (ap1 + bp1)/2 ) / (ap1 - bp1) * volume
        vdiff = vdiff.rolling(n).sum()
        volume = volume.rolling(n).sum()
        vdiff = (vdiff / volume).fillna(0)

        return vdiff



