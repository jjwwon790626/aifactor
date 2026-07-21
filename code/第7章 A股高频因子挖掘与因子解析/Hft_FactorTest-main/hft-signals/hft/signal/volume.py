from .IFactor import IFactor

class ask_volume(IFactor):

    describe = {
        "name": "ask_volume",
        "datas": [
            "depth5"
        ],
        "params": [],
        "description": """
        ask volume.
        """
    }
    
    def main(cls, **kwargs):
        depth5 = kwargs.get("datas").get("depth5")
        return depth5.av1

class bid_volume(IFactor):

    describe = {
        "name": "bid_volume",
        "datas": [
            "depth5"
        ],
        "params": [],
        "description": """
        bid volume.
        """
    }
    
    def main(cls, **kwargs):
        depth5 = kwargs.get("datas").get("depth5")
        return depth5.bv1