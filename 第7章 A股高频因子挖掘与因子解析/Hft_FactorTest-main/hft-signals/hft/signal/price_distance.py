from .IFactor import IFactor

class price_distance(IFactor):

    describe = {
        "name": "price_distance",
        "datas": [
            "depth5"
        ],
        "params": [
        {
            "name": "side",
            "default_value": "bid"
        },
        {
            "name": "n",
            "default_value": 100
        }],
        "description": """
        price all passing by distance.
        """
    }
    
    def main(cls, **kwargs):
        depth5 = kwargs.get("datas").get("depth5")
        side = kwargs.get("params").get("side")
        n = kwargs.get("params").get("n")
        
        if side == "ask":
            prices = depth5.ap1
        else:
            prices = depth5.bp1

        p_change = prices.diff().abs().rolling(n).sum().fillna(0)
        return p_change.rename(f'price_distance_{side}_{n}ticks')