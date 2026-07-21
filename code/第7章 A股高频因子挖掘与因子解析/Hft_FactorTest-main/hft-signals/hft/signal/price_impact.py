from .IFactor import IFactor
import pandas as pd

class price_impact(IFactor):

    describe = {
        "name": "price_impact",
        "datas": [
            "depth5"
        ],
        "params": [{
            "name": "n",
            "default_value": 5
        }],
        "description": """
        Weak Orderbook Imbalance Ratio.
        """
    }
    
    def main(cls, **kwargs):
        depth5 = kwargs.get("datas").get("depth5")
        n = kwargs.get("params").get("n")

        ask, bid, ask_v, bid_v = 0, 0, 0, 0
        for i in range(1, n+1):
            ask += depth5[f'ap{i}'] * depth5[f'av{i}']
            bid += depth5[f'bp{i}'] * depth5[f'bv{i}']
            ask_v += depth5[f'av{i}']
            bid_v += depth5[f'bv{i}']
        ask /= ask_v
        bid /= bid_v
        return pd.Series(-(depth5['ap1'] - ask)/depth5['ap1'] - (depth5['bp1'] - bid)/depth5['bp1'], name="price_impact")