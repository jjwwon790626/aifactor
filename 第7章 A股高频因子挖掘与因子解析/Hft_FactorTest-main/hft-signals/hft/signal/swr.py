from .IFactor import IFactor
import numpy as np

class swr(IFactor):

    describe = {
        "name": "swr",
        "datas": [
            "depth5"
        ],
        "params": [
            {
                "name": "side",
                "default_value": "bid"
            }
        ],
        "description": """
        Orderbook Imbalance Ratio.
        """
    }
    
    def main(cls, **kwargs):
        depth5 = kwargs.get("datas").get("depth5")
        side = kwargs.get("params").get("side")

        if side == 'bid':
            return depth5.bv1 / (depth5.bv2 + depth5.bv3 + depth5.bv4 + depth5.bv5)
        elif side == 'ask':
            return depth5.av1 / (depth5.av2 + depth5.av3 + depth5.av4 + depth5.av5)