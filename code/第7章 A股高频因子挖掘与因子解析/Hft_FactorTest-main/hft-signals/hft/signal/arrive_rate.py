from .IFactor import IFactor
from hft.utils.wrapper import trade_to_depth
import pandas as pd

class arrive_rate(IFactor):

    describe = {
        "name": "arrive_rate",
        "datas": [
            "depth5",
            "trade"
        ],
        "params": [
            {
                "name": "n",
                "default_value": 100,
            }
        ],
        "description": """
        arrive_rate.
        """
    }
    
    @trade_to_depth
    def main(cls, **kwargs):
        trade = kwargs['datas']['trade']
        n = kwargs['params']['n']
        
        res = trade['ts'].diff(n).fillna(0) / n 
        return res



