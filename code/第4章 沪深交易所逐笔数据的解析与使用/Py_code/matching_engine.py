from datetime import datetime

from order_matching.executed_trades import ExecutedTrades
from order_matching.order import Order
from order_matching.order_book import OrderBook
from order_matching.orders import Orders
from order_matching.random import get_faker
from order_matching.status import Status
from order_matching.trade import Trade

# 这两个是为了对不同类型的订单进行处理
from order_matching.execution import Execution
from order_matching.side import Side


class MatchingEngine:
    """Order Book Matching Engine.

    Parameters
    ----------
    seed
        Random seed

    Examples
    --------
    >>> from datetime import datetime, timedelta
    >>> from pprint import pp
    >>> from order_matching.matching_engine import MatchingEngine
    >>> from order_matching.order import LimitOrder
    >>> from order_matching.side import Side
    >>> matching_engine = MatchingEngine(seed=123)
    >>> timestamp = datetime(2023, 1, 1)
    >>> transaction_timestamp = timestamp + timedelta(days=1)
    >>> buy_order = LimitOrder(side=Side.BUY, price=1.2, size=2.3, timestamp=timestamp, order_id="a", trader_id="x")
    >>> sell_order = LimitOrder(side=Side.SELL, price=0.8, size=1.6, timestamp=timestamp, order_id="b", trader_id="y")
    >>> executed_trades = matching_engine.match(orders=Orders([buy_order, sell_order]), timestamp=transaction_timestamp)
    >>> pp(executed_trades.trades)
    [Trade(side=SELL,
           price=1.2,
           size=1.6,
           incoming_order_id='b',
           book_order_id='a',
           execution=LIMIT,
           trade_id='c4da537c-1651-4dae-8486-7db30d67b366',
           timestamp=datetime.datetime(2023, 1, 2, 0, 0))]
    """

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self._faker = get_faker(seed=seed)
        self._queue = Orders()
        self.unprocessed_orders = OrderBook()
        self._timestamp: datetime





    def match(self, timestamp: datetime, orders: Orders | None = None) -> ExecutedTrades:
        """Match incoming orders in price-time priority.

        Parameters
        ----------
        timestamp
            Timestamp of order matching
        orders
            Incoming orders. Will be matched with existing ones on the order book in

        Returns
        -------
        ExecutedTrades
            Executed trades storage object
        """
        self._timestamp = timestamp
        self._queue += orders if orders else Orders()
        self._queue += self._get_expired_orders()
        trades = ExecutedTrades()
        while not self._queue.is_empty:
            trades += self._match(order=self._queue.dequeue())
        return trades
    
    def get_snapshot(self) -> OrderBook:
        '''
        To reveal the current snapshot
        Returns
        -------
        OrderBook
            orderbook.

        '''
        # orders not executed yet
        unprocessed_orders = self.unprocessed_orders     
        return unprocessed_orders

        '''
        To reveal the current snapshot
        Returns
        -------
        OrderBook
            orderbook.

        '''
    @property
    def snapshot(self) -> OrderBook:
        return self.unprocessed_orders


    def cancel_order(self, orderID: str):
        '''
        Parameters
        ----------
        orderID : str
            cancel a list of order, order ID is string

        Returns
        -------
        No return

        '''
        orderID = str(orderID)
        for price in list(self.unprocessed_orders.bids.keys()):
            for order in list(self.unprocessed_orders.bids[price]):
                if order.order_id == orderID:
                    order.status = Status.CANCEL
                    self.unprocessed_orders.remove(incoming_order=order)



                
        for price in list(self.unprocessed_orders.offers.keys()):
            for order in list(self.unprocessed_orders.offers[price]):
                if order.order_id == orderID:
                    order.status = Status.CANCEL
                    self.unprocessed_orders.remove(incoming_order=order)
                

    def _get_expired_orders(self) -> Orders:
        orders: list[Order] = list()
        for timestamp in filter(lambda t: t <= self._timestamp, self.unprocessed_orders.orders_by_expiration.keys()):
            orders.extend(self.unprocessed_orders.orders_by_expiration[timestamp])
        for order in orders:
            order.status = Status.CANCEL
        return Orders(orders)

    def _match(self, order: Order) -> ExecutedTrades:

        match order.execution:
            case Execution.LIMIT:   
                # 这是对于限价单的处理办法        
                if order.status == Status.CANCEL:
                    self.unprocessed_orders.remove(incoming_order=order)
                    return ExecutedTrades()
                elif self.unprocessed_orders.matching_order_exists(incoming_order=order):
                    return self._execute_trades(incoming_order=order)
                else:
                    self.unprocessed_orders.append(incoming_order=order)
                    return ExecutedTrades()


                
            case Execution.MARKETLIMITORDER:
                #（一）对手方最优价格申报
                # 优先查看订单簿中的对手方最优的价格
                if order.get_side()==Side.SELL:
                    # 如果是卖单查看对手方买单
                    bestOpposite = self.get_snapshot().max_bid

                elif order.get_side()==Side.BUY:
                    bestOpposite = self.get_snapshot().min_offer

                order.set_price(bestOpposite)
                # 对手最优需要输入side,不需要price，这个直到遇到了orderbook要处理才需要输入price,size是订单的大小,timestamp是初始化的；
                if order.status == Status.CANCEL:
                    self.unprocessed_orders.remove(incoming_order=order)
                    return ExecutedTrades()                  

                elif self.unprocessed_orders.matching_order_exists(incoming_order=order):
                    return self._execute_trades(incoming_order=order)
                else:
                    # 对手方市价申报类型进入交易主机时，集中申报簿中对手方无申报的，申报自动撤销。
                    order.status = Status.CANCEL
                    self.unprocessed_orders.remove(incoming_order=order)
                    return ExecutedTrades()                

            case Execution.BOP:
                #（二）本方最优价格申报；  这个已经处理了（Best of Party）
                # 本方最优就优先查看订单簿中的本方最优的价格
                if order.get_side()==Side.BUY:
                    bestBid = self.get_snapshot().max_bid
                    if bestBid>0:
                       order.set_price(bestBid)
                    else:
                       # 如果本方没有订单，直接处理掉，设置为cancel
                       order.status=Status.CANCEL
                elif order.get_side()==Side.SELL:
                    bestOffer = self.get_snapshot().min_offer
                    if bestOffer<float("inf"):
                        order.set_price(bestOffer)
                    else:
                        order.status=Status.CANCEL
                # 本方最优需要输入side,不需要price，这个直到遇到了orderbook要处理才需要输入price,size是订单的大小,timestamp是初始化的；
                if order.status == Status.CANCEL:
                    self.unprocessed_orders.remove(incoming_order=order)
                    return ExecutedTrades()                  

                #elif self.unprocessed_orders.matching_order_exists(incoming_order=order):
                #    return self._execute_trades(incoming_order=order)
                else:
                    self.unprocessed_orders.append(incoming_order=order)
                    return ExecutedTrades()

            case Execution.FAK5L:
                if order.get_side()==Side.BUY:
                    oppositeSitePrice = self.get_snapshot()._get_offer_prices()

                elif order.get_side()==Side.SELL:
                    oppositeSitePrice = self.get_snapshot()._get_bid_prices()
                if len(oppositeSitePrice)>=5:
                    price5L = oppositeSitePrice[4]
                elif len(oppositeSitePrice)>0:
                    price5L = oppositeSitePrice[-1]
                elif len(oppositeSitePrice) == 0:
                    if order.get_side()==Side.BUY:
                        price5L = 0
                    else:
                        price5L = float("inf")

                # 直接按照5L价格下单，即,取消的逻辑会在没有匹配对手方订单的逻辑里面处理
                order.set_price(price5L)
                if order.status == Status.CANCEL:
                    self.unprocessed_orders.remove(incoming_order=order)
                    return ExecutedTrades()                  

                elif self.unprocessed_orders.matching_order_exists(incoming_order=order):
                    return self._execute_trades(incoming_order=order)
                else:
                    # 如果找不到匹配的，直接kill订单
                    order.status = Status.CANCEL
                    self.unprocessed_orders.remove(incoming_order=order)
                    return ExecutedTrades()                






            case Execution.FAKA:
                # 因为买单的FAKA设置了强行买价无穷，卖价为0，所以如果成交不到就会自动撤单，能成就会全部成交
                if order.status == Status.CANCEL:
                    self.unprocessed_orders.remove(incoming_order=order)
                    return ExecutedTrades()
                elif self.unprocessed_orders.matching_order_exists(incoming_order=order):
                    return self._execute_trades(incoming_order=order)
                else:
                    order.status = Status.CANCEL
                    self.unprocessed_orders.remove(incoming_order=order)
                    return ExecutedTrades()

                    
                    
                
            case Execution.FOKA:
                if order.get_side()==Side.BUY:
                    oppositeSiteSize = self.get_snapshot()._get_offer_sizes()

                elif order.get_side()==Side.SELL:
                    oppositeSiteSize = self.get_snapshot()._get_bid_sizes()
                    
                if len(oppositeSiteSize)>0:
                    oppositeSiteSizeSum = sum(oppositeSiteSize)
                else:
                    oppositeSiteSizeSum = 0
                # 如果发出的订单大于所有，那么记得要cancel剩余订单
                if order.size >oppositeSiteSizeSum:
                    order.status = Status.CANCEL

                    self.unprocessed_orders.remove(incoming_order=order)
                    return ExecutedTrades()                  
    
                     # 如果订单数量小于等于所有，那么不用cancel，因为全部会成交
                else:
                    if order.status == Status.CANCEL:
                        self.unprocessed_orders.remove(incoming_order=order)
                        return ExecutedTrades()                  
    
                    elif self.unprocessed_orders.matching_order_exists(incoming_order=order):
 
                        return self._execute_trades(incoming_order=order)
                    else:
                        order.status = Status.CANCEL
                        self.unprocessed_orders.remove(incoming_order=order)
                        return ExecutedTrades()       
                        

    def _execute_trades(self, incoming_order: Order) -> ExecutedTrades:
        trades = ExecutedTrades()
        for price in self.unprocessed_orders.get_matching_sorted_opposite_side_prices(incoming_order=incoming_order):

            trades += self._execute_trades_for_one_price(incoming_order=incoming_order, price=price)
        if incoming_order.size > 0:
            if incoming_order.execution in [Execution.FAK5L,Execution.FAKA]:
                incoming_order.status = Status.CANCEL
                self.unprocessed_orders.remove(incoming_order=incoming_order)
            else:
                self.unprocessed_orders.append(incoming_order=incoming_order)
        return trades

    def _execute_trades_for_one_price(self, incoming_order: Order, price: float) -> ExecutedTrades:
        opposite_side_orders = self.unprocessed_orders.get_opposite_side_orders(incoming_order=incoming_order)
        trades, zero_size_orders = list(), list()
        for book_order in opposite_side_orders[price]:
            if incoming_order.size > 0:
                trades.append(self._execute_trade(incoming_order=incoming_order, book_order=book_order))
            if book_order.size == 0:

                zero_size_orders.append(book_order)
        opposite_side_orders[price].remove(orders=zero_size_orders)
        if len(list(filter(lambda order: order.size > 0, opposite_side_orders[price]))) == 0:
            opposite_side_orders.pop(price)
        return ExecutedTrades(trades=trades)

    def _execute_trade(self, incoming_order: Order, book_order: Order) -> Trade:
        trade = Trade(
            side=incoming_order.side,
            price=book_order.price,
            size=min(incoming_order.size, book_order.size),
            incoming_order_id=incoming_order.order_id,
            book_order_id=book_order.order_id,
            timestamp=self._timestamp,
            execution=incoming_order.execution,
            trade_id=self._faker.uuid4(),
        )
        incoming_order.size = max(0.0, incoming_order.size - trade.size)
        book_order.size = max(0.0, book_order.size - trade.size)
        return trade
    
    def get_all_orders(self) -> Orders:
        """获取所有订单并返回一个 Orders 对象。"""
        all_orders = Orders()  # 创建一个新的 Orders 对象
        
        # 遍历所有买单
        bid_number,offer_number = 0,0
        for orders in self.unprocessed_orders.bids.values():
            for order in orders.orders:
                all_orders.add([order])
                bid_number+=1

        # 遍历所有卖单
        for orders in self.unprocessed_orders.offers.values():
            for order in orders.orders:
                all_orders.add([order])
                offer_number+=1
                
        print('retrieved {} orders from the orderbook, while the orderbook has {} bids, {} offers'.format(len(all_orders),bid_number,offer_number))

        return all_orders