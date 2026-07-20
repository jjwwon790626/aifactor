from order_matching.custom_enum import CustomEnum


class Execution(CustomEnum):
    """Order execution."""

    MARKET = 0
    LIMIT = 1
    
#（一）对手方最优价格申报；最终只会有一种价格，且只有成交，或者撤单，需要检查订单簿  （MarketLimitOrder）
#（二）本方最优价格申报；  这个已经处理了 (Best of Party    BOP)
#（三）最优五档即时成交剩余撤销申报；   最终会有多种价格且会撤单  (FAK5L)
#（四）即时成交剩余撤销申报；   可能有多种价格，且有可能会有撤销    (FAKA)
#（五）全额成交或撤销申报；  可能有多种价格，或者撤销 （FOKA）

    MARKETLIMITORDER = 2
    BOP = 3
    FAK5L = 4
    FAKA = 5
    FOKA = 6