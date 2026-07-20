from .IFactor import IFactor

class weakoir(IFactor):

    describe = {
        "name": "weakoir",
        "datas": [
            "depth5"
        ],
        "params": [],
        "description": """
        Weak Orderbook Imbalance Ratio.
        """
    }
    
    def main(cls, **kwargs):
        depth5 = kwargs.get("datas").get("depth5")

        bid_volumes = depth5["bv2"] + depth5["bv3"] + depth5["bv4"] + depth5["bv5"]
        ask_volumes = depth5["av2"] + depth5["av3"] + depth5["av4"] + depth5["av5"]

        return 2 * bid_volumes / (bid_volumes + ask_volumes) - 1 