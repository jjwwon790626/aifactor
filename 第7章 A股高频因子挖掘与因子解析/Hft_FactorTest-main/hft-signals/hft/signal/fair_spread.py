from .IFactor import IFactor
import pandas as pd

class fair_spread(IFactor):

    describe = {
        "name": "fair_spread",
        "datas": [
            "depth5"
        ],
        "params": [],
        "description": """
        fair_spread.
        """
    }
    
    def main(cls, **kwargs):
        depth5 = kwargs.get("datas").get("depth5")
        ob = depth5
        res = (ob['bp2']*ob['bv2'] + ob['bp3']*ob['bv3'] + ob['bp4']*ob['bv4'] + ob['bp5']*ob['bv5'])/(ob['bv2']+ob['bv3']+ob['bv4']+ob['bv5']) + (ob['ap2']*ob['av2'] + ob['ap3']*ob['av3'] + ob['ap4']*ob['av4'] + ob['ap5']*ob['av5'])/(ob['av2']+ob['av3']+ob['av4']+ob['av5']) - (ob['ap1']+ob['bp1'])
        return pd.Series(res, index=ob.index, name='fair_spread')