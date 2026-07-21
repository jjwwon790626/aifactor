from .IFactor import IFactor
import numpy as np
import numba as nb
from hft.utils.wrapper import trade_to_depth
class llt(IFactor):

    describe = {
        "name": "llt",
        "datas": [
            "depth5",
        ],
        "params": [{
            "name": "n",
            "default_value": 100
        }
        ],
        "description": """
        fast fast moving avg
        """
    }
    
    @staticmethod
    @nb.jit
    def _LLT(trade, par):
        a = 2/(par+1)
        a2 = a*a
        # cp = np.array(trade)
        # tmp = np.array(trade)
        cp = trade.copy()
        tmp = trade.copy()
        for i in np.arange(2, len(trade)):
            tmp[i] = (a-(a2)/4)*cp[i]+((a2)/2)*cp[i-1] - (a-3*(a2)/4)*cp[i-2]+2*(1-a)*tmp[i-1]-(1-a)*(1-a)*tmp[i-2]
        return tmp

    def main(cls, **kwargs):
        depth5 = kwargs.get("datas").get("depth5")
        n = kwargs.get("params", {}).get("n", 100)

        _mp = ((depth5.bp1 + depth5.ap1)/2).values
        
        _llt = cls._LLT(_mp, n)
        return _llt

class llt_trend(IFactor):
    
    describe = {
        "name": "llt_trend",
        "datas": [
            "depth5",
        ],
        "params": [
            {
                "name": "n",
                "default_value": 100
            },
            {
                "name": "filt",
                "default_value": "all"
            }
        ],
        "description": """
        fast fast moving avg
        """
    }

    def llt_uptrend(llt_series):
        llt_series = pd.Series(llt_series)
        return (llt_series > llt.series.shift(1)).fillna(0)

    def llt_downtrend(llt_series):
        llt_series = pd.Series(llt_series)
        return (llt_series < llt_series.shift(1)).fillna(0)

    def llt_trend(llt_series):
        return np.sign(llt_series.diff().fillna(0))

    def main(cls, **kwargs):
        depth5 = kwargs.get("datas").get("depth5")
        n = kwargs.get("params", {}).get("n", 100)
        filt = kwargs.get("params", {}).get("filt", "all")

        _llt = llt(kwargs)

        if filt == "all":
            return cls.llt_trend(_llt)
        elif filt == "up":
            return cls.llt_uptrend(_llt)
        elif filt == "down":
            return cls.llt_downtrend(_llt)

        return _llt

class llt_trend_start(IFactor):
    
    describe = {
        "name": "llt_trend_start",
        "datas": [
            "depth5",
        ],
        "params": [
            {
                "name": "n",
                "default_value": 100
            },
            {
                "name": "filt",
                "default_value": "all"
            }
        ],
        "description": """
        fast fast moving avg trend start
        """
    }

    def llt_uptrend_start(cls, llt_series):
        llt_series = np.sign(np.sign(pd.Series(llt_series).diff()).diff()).fillna(0)
        return (llt_series > 0).astype(int)
        
    def llt_downtrend_start(cls, llt_series):
        llt_series = np.sign(np.sign(pd.Series(llt_series).diff()).diff()).fillna(0)
        return (llt_series < 0).astype(int)

    def llt_trend_start(cls, llt_series):
        return np.sign(np.sign(pd.Series(llt_series).diff()).diff()).fillna(0)

    def main(cls, **kwargs):
        depth5 = kwargs.get("datas").get("depth5")
        n = kwargs.get("params", {}).get("n", 100)
        filt = kwargs.get("params", {}).get("filt", "all")

        _llt = llt(kwargs)

        if filt == "all":
            return cls.llt_trend_start(_llt)
        elif filt == "up":
            return cls.llt_uptrend_start(_llt)
        elif filt == "down":
            return cls.llt_downtrend_start(_llt)

        return _llt
