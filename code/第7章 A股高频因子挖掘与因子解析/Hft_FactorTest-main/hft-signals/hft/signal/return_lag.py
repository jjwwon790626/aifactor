from .IFactor import IFactor

class ask_lag(IFactor):

    describe = {
        "name": "ask_lag",
        "datas": [
            "depth5"
        ],
        "params": [{
            "name": "n",
            "default_value": 100
        }],
        "description": """
        ask volume.
        """
    }
    
    def main(cls, **kwargs):
        depth5 = kwargs.get("datas").get("depth5")
        n = kwargs.get("params", {}).get("n")
        return depth5.ap1.diff(n).fillna(0)

class bid_lag(IFactor):

    describe = {
        "name": "bid_lag",
        "datas": [
            "depth5"
        ],
        "params": [{
            "name": "n",
            "default_value": 100
        }],
        "description": """
        bid volume.
        """
    }
    
    def main(cls, **kwargs):
        depth5 = kwargs.get("datas").get("depth5")
        n = kwargs.get("params", {}).get("n")
        return depth5.bp1.diff(n).fillna(0)