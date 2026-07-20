from .IFactor import IFactor

class large_jump(IFactor):

    describe = {
        "name": "large_jump",
        "datas": [
            "depth5"
        ],
        "params": [{
            "name": "jump_ratio",
            "default_value": 0.002,
        },
        {
            "name": "direct",
            "default_value": "up"
        },
        {
            "name": "side",
            "default_value": "bid"
        },
        {
            "name": "n",
            "default_value": 100
        }],
        "description": """
        large jump.
        """
    }
    
    def main(cls, **kwargs):
        depth5 = kwargs.get("datas").get("depth5")
        direct = kwargs.get("params").get("direct")
        side = kwargs.get("params").get("side")
        n = kwargs.get("params").get("n")
        jump_ratio = kwargs.get("params").get("jump_ratio")
        
        if side == "ask":
            prices = depth5.ap1
        else:
            prices = depth5.bp1

        p_change = prices.diff(n) / prices.shift(n)

        if direct == "down":
            result = p_change < -jump_ratio
        elif direct == "up":
            result = p_change > jump_ratio

        return result.rename(f'large_jump_{direct}_{jump_ratio}')