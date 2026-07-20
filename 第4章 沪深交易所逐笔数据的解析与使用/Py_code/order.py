from dataclasses import dataclass, field
from datetime import datetime

from order_matching.execution import Execution
from order_matching.side import Side
from order_matching.status import Status


@dataclass(kw_only=True)
class Order:
    """Single order base storage class."""

    side: Side
    price: float
    size: float
    timestamp: datetime
    order_id: str
    trader_id: str
    execution: Execution
    expiration: datetime = datetime.max
    status: Status = Status.OPEN
    price_number_of_digits: int = 3

    def __post_init__(self) -> None:
        self.price = round(number=self.price, ndigits=self.price_number_of_digits)
        self.order_id = str(self.order_id)
        
        
    def get_side(self) -> Side:
        return self.side
        
    def get_price(self) -> float:
        return self.price
    
    def set_price(self, priceToBeSet:float) -> None:
        self.price = priceToBeSet
        self.price = round(number=self.price, ndigits=self.price_number_of_digits)
        
        # 对于那些fak 订单，需要指定到第五档的价格，这个时候如果能成交完，就不用撤单；如果成交不完，就需要撤单;
        
    def get_size(self) -> float:
        # 获取订单剩余的大小
        return self.size


@dataclass(kw_only=True)
class LimitOrder(Order):
    """Single limit order storage class."""

    execution: Execution = field(init=False, default=Execution.LIMIT)


@dataclass(kw_only=True)
class MarketOrder(Order):
    """Single market order storage class."""

    execution: Execution = field(init=False, default=Execution.MARKET)
    price: float = field(init=False)

    def __post_init__(self) -> None:
        self.price = 0 if self.side == Side.SELL else float("inf")
        self.order_id = str(self.order_id)
        

# 如果是市价的打对手盘总共有几种情况
#（一）对手方最优价格申报；最终只会有一种价格，且只有成交，或者撤单，需要检查订单簿  （MarketLimitOrder）
#（二）本方最优价格申报；  这个已经处理了（Best of Party）
#（三）最优五档即时成交剩余撤销申报；   最终会有多种价格且会撤单  (FAK5L)
#（四）即时成交剩余撤销申报；   可能有多种价格，且有可能会有撤销    (FAKA)
#（五）全额成交或撤销申报；  可能有多种价格，或者撤销 （FOKA）
# 可以根据盘口来测算，如果是对手盘最优价格申报，就转化为一个一开始设置的定量的限价单，如果不存在就取消
# 最优五档则是需要读取对手盘最优五档盘口的挂单量，如果0档就直接撤销，一直往上数所有的非空档位，然后发送数量和价格，价格取第五档或者在不满五档时取最大
# 下的订单的价格是五档以外取五档，五档以内取买单就取卖价最高，卖单就取买价最低
# 直到成交为止那么需要确定的是这五档中哪一档为需要发送的价格，只需要确定价格，还需要确定min(直到这一档的总量，和订单的总量)即可
# 即时成交剩余撤销，取的是第一档盘口，数量则是发送的数量，剩余的是一个撤销订单，可以把这两个
# 把前五档的数量加起来，和订单数量比较，如果没有5档，就把有的数量加起来，如果大于等于那个
# 市价单数量，就发一个对应的价格的限价单；如果小于那个市价单数量，数量就取能取到的限价订单簿的数量，然后剩余订单直接撤销

# Market Limit Order - Market Limit orders are executed at the best available price on the opposite side of the market. If the order cannot be completely filled, the order becomes a Limit order and the remaining quantity rests on the order book at the fill price. If no market is available, the order is rejected.

# 本方最优价格申报进入交易主机时，集中申报簿中
# 本方无申报的，申报自动撤销。
# 其他市价申报类型进入交易主机时，集中申报簿中对手方无
# 申报的，申报自动撤销。

# 订单类应当只包含订单类型，而至于如何处理这些订单，应当写在matchingEngine里面


@dataclass(kw_only=True)
class MarketLimitOrder(Order):
    """Single marketlimit order storage class."""

    execution: Execution = field(init=False, default=Execution.MARKETLIMITORDER)
    price: float = field(init=False)

    def __post_init__(self) -> None:
        self.price = 0 if self.side == Side.SELL else float("inf")
        self.order_id = str(self.order_id)
        
        #（一）对手方最优价格申报；最终只会有一种价格，且只有成交，或者撤单，需要检查订单簿  （MarketLimitOrder）
        # 下这种订单，只需要确定方向，不需要传入订单簿，但处理的时候需要传入订单簿
        

@dataclass(kw_only=True)
class BestOfPartyOrder(Order):
    """Single best of party order storage class."""

    execution: Execution = field(init=False, default=Execution.BOP)
    price: float = field(init=False)

    def __post_init__(self) -> None:
        self.price = 0 if self.side == Side.BUY else float("inf")
        self.order_id = str(self.order_id)
        #（二）本方最优价格申报；  这个已经处理了（Best of Party）
        
        # 本方最优需要输入side,不需要price，这个直到遇到了orderbook要处理才需要输入price,size是订单的大小,timestamp是初始化的；

    

        
@dataclass(kw_only=True)
class FAK5LOrder(Order):
    """Single fak5l order storage class."""
    
    # （三）最优五档即时成交剩余撤销申报；   最终会有多种价格且会撤单  (FAK5L)

    execution: Execution = field(init=False, default=Execution.FAK5L)
    price: float = field(init=False)

    def __post_init__(self) -> None:
        self.price = 0 if self.side == Side.SELL else float("inf")
        self.order_id = str(self.order_id)
        
@dataclass(kw_only=True)
class FAKAOrder(Order):
    """Single FAKA order storage class."""
    
    # （四）即时成交剩余撤销申报；   可能有多种价格，且有可能会有撤销    (FAKA)

    execution: Execution = field(init=False, default=Execution.FAKA)
    price: float = field(init=False)

    def __post_init__(self) -> None:
        self.price = 0 if self.side == Side.SELL else float("inf")
        self.order_id = str(self.order_id)
        
        
        
@dataclass(kw_only=True)
class FOKAOrder(Order):
    """Single FOKA order storage class."""
    
    # 全额成交或撤销申报；  可能有多种价格，或者撤销 （FOKA）

    execution: Execution = field(init=False, default=Execution.FOKA)
    price: float = field(init=False)

    def __post_init__(self) -> None:
        self.price = 0 if self.side == Side.SELL else float("inf")
        self.order_id = str(self.order_id)
        