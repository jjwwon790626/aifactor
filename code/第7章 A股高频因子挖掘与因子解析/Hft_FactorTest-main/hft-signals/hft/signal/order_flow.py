from hft.signal.IFactor import IFactor
import numpy as np
import numba as nb

@nb.njit
def bid_order_flow(l, n):
    
    last_bid_1_price = l[0] # bp1
    last_bid_5_price = l[-6] # bp5
    
    V = l[1] + l[5] + l[9] + l[13] + l[17]
    #   bv1  + bv2  + bv3  + bv4   + bv5

    for price_index in range(0,19,4):
        this_price = n[price_index]
        this_volume = n[price_index+1]
        
        if last_bid_1_price < this_price:
            # new bid
            V += this_volume
    
        elif last_bid_5_price > this_price:
            # pass
            V *= 0.8
            V += this_volume
        else: 
            V -= this_volume
    return V

@nb.njit
def ask_order_flow(l, n):
    
    last_ask_1_price = l[2] # ap1
    last_ask_5_price = l[-4] # ap5
    
    V = l[3] + l[7] + l[11] + l[15] + l[19]
    #   av1  + av2  + av3   + av4   + av5
    for price_index in range(2,21,4):
        # ap1 -> ap5
        this_price = n[price_index]
        this_volume = n[price_index+1]
        
        if last_ask_1_price > this_price:
            # new bid
            V += this_volume
    
        elif last_ask_5_price < this_price:
            # pass
            V *= 0.8
            V += this_volume
        else:
            V -= this_volume
    return V

class oflow(IFactor):

    describe = {
        "name": "oflow",
        "datas": [
            "depth5"
        ],
        "params": [
            {
                "name": "side",
                "default_value": "ask"
            },
            {
                "name": "bend_ratio",
                "default_value": 4
            }
        ],
        "description": """
        Order Flow.
        """
    }
    
    # def main(cls, **kwargs):
    #     depth5 = kwargs.get("datas").get("depth5")

    #     bend_ratio = kwargs.get("params", {}).get("bend_ratio")
    #     if bend_ratio is None:
    #         bend_ratio = 4

    #     if kwargs.get("params").get("side") == 'bid':
    #         ob_values = depth5.values.astype(np.float64)
    #         flows = np.zeros(len(ob_values))
    #         for i in range(1, len(ob_values)):
    #             flows[i] = bid_order_flow(ob_values[i-1], ob_values[i])
    #     elif kwargs.get("params").get("side") == 'ask':
    #         ob_values = depth5.values.astype(np.float64)
    #         flows = np.zeros(len(ob_values))
    #         for i in range(1, len(ob_values)):
    #             flows[i] = ask_order_flow(ob_values[i-1], ob_values[i])

    #     flows[flows < 0] *= -bend_ratio
        
    #     return flows

    def main(cls, **kwargs):
        depth5 = kwargs.get("datas").get("depth5")
        
        bend_ratio = kwargs.get("params", {}).get("bend_ratio")
        if bend_ratio is None:
            bend_ratio = 4
        
        # 只选择价格和数量列（从第5列开始，即 ap1 开始）
        price_volume_cols = ['bp1', 'bv1', 'bp2', 'bv2', 'bp3', 'bv3', 'bp4', 'bv4', 'bp5', 'bv5',
                            'ap1', 'av1', 'ap2', 'av2', 'ap3', 'av3', 'ap4', 'av4', 'ap5', 'av5']
        
        # 获取价格和数量数据
        ob_values = depth5[price_volume_cols].values.astype(np.float64)
        flows = np.zeros(len(ob_values))
        
        if kwargs.get("params").get("side") == 'bid':
            for i in range(1, len(ob_values)):
                flows[i] = bid_order_flow(ob_values[i-1], ob_values[i])
        elif kwargs.get("params").get("side") == 'ask':
            for i in range(1, len(ob_values)):
                flows[i] = ask_order_flow(ob_values[i-1], ob_values[i])
        
        flows[flows < 0] *= -bend_ratio
        
        import pandas as pd
        return pd.Series(flows, index=depth5.index)
    