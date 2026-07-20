from .IFactor import IFactor

class OIR(IFactor):

    describe = {
        "name": "OIR",
        "datas": [
            "depth5"
        ],
        "params": [],
        "description": """
        Orderbook Imbalance Ratio.
        """
    }
    
    def main(cls, **kwargs):
        depth5 = kwargs.get("datas").get("depth5")
        return 2 * depth5['bv1'] / (depth5['av1'] + depth5['bv1']) - 1 