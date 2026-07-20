from dataclasses import dataclass
from datetime import datetime

from order_matching.execution import Execution
from order_matching.side import Side
# from pandera.typing import DateTime, Series


@dataclass(kw_only=True)
class Trade:
    """Single trade storage class."""

    side: Side
    price: float
    size: float
    incoming_order_id: str
    book_order_id: str
    execution: Execution
    trade_id: str
    timestamp: datetime
    

  
    # def initSeries(self,trade:Series):

    #     self.side = Side.BUY if trade['TradeBSFlag_T'] == 'B' else Side().SELL
    #     self.price = trade['TradePrice_T']
    #     self.size = trade['TradeQty_T']
    #     self.incoming_order_id = str(int(max(trade['BidApplSeqNum_T'],trade['OfferApplSeqNum_T'])))
    #     self.book_order_id = str(int(min(trade['BidApplSeqNum_T'],trade['OfferApplSeqNum_T'])))
    #     self.execution = Execution.LIMIT
    #     self.trade_id = None
    #     self.timestamp = trade['TradeTime']
        