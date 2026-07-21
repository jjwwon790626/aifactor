from .IFactor import IFactor
import numpy as np

class oir(IFactor):

    describe = {
        "name": "oir",
        "datas": [
            "depth5"
        ],
        "params": [
            {
                "name": "log",
                "default_value": True
            }
        ],
        "description": """
        Orderbook Imbalance Ratio.
        """
    }
    
    def main(cls, **kwargs):
        depth5 = kwargs.get("datas").get("depth5")
        if kwargs.get("params",{}).get("log"):
            return 2 * np.log(depth5['bv1'] + 1) / (np.log(depth5['av1'] + 1) + np.log(depth5['bv1'] + 1)) - 1
        else:
            return 2 * depth5['bv1'] / (depth5['av1'] + depth5['bv1']) - 1 
        
# 有窗口版      
# class oir_with_window(IFactor):
#     describe = {
#         "name": "oir_with_window",
#         "datas": ["depth5"],
#         "params": [
#             {
#                 "name": "log",
#                 "default_value": True
#             },
#             {
#                 "name": "window",  # 添加窗口参数
#                 "default_value": 1
#             }
#         ]
#     }
    
#     def main(cls, **kwargs):
#         depth5 = kwargs.get("datas").get("depth5")
#         window = kwargs.get("params", {}).get("window", 1)
#         log_flag = kwargs.get("params", {}).get("log", True)
        
#         if log_flag:
#             oir_raw = 2 * np.log(depth5['bv1'] + 1) / (np.log(depth5['av1'] + 1) + np.log(depth5['bv1'] + 1)) - 1
#         else:
#             oir_raw = 2 * depth5['bv1'] / (depth5['av1'] + depth5['bv1']) - 1
            
#         # 应用窗口
#         if window > 1:
#             return oir_raw.rolling(window).mean()
#         else:
#             return oir_raw