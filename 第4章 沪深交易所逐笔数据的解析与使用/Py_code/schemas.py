from pandera import DataFrameModel, Field
from pandera.typing import DateTime, Series
import ast

from order_matching.execution import Execution
from order_matching.side import Side
from order_matching.status import Status



class BaseOrderSchema(DataFrameModel):
    """Base order schema."""

    side: Series[str] = Field(isin=[Side.BUY.name, Side.SELL.name])
    price: Series[float] = Field(gt=0)
    size: Series[float] = Field(gt=0)


class OrderBookSummarySchema(BaseOrderSchema):
    """Order book summary schema."""

    price: Series[float] = Field(unique=True, gt=0)
    count: Series[int] = Field(ge=0)

    class Config:
        strict = True


class OrderDataSchema(BaseOrderSchema):
    """Order data schema."""

    timestamp: Series[DateTime]
    expiration: Series[DateTime] = Field(nullable=True)
    order_id: Series[str]
    trader_id: Series[str]
    execution: Series[str] = Field(isin=[Execution.MARKET.name, Execution.LIMIT.name])
    status: Series[str] = Field(isin=[Status.OPEN.name, Status.CANCEL.name])
    price_number_of_digits: Series[int]

    class Config:
        strict = True


class TradeDataSchema(BaseOrderSchema):
    """Trade data schema."""

    timestamp: Series[DateTime]
    incoming_order_id: Series[str]
    book_order_id: Series[str]
    trade_id: Series[str]
    execution: Series[str] = Field(isin=[Execution.MARKET.name, Execution.LIMIT.name])

    class Config:
        strict = True


# class SnapshotSchema():
#     """Snapshot schema."""
#     price_level: tuple(str,float) = Field(isin)# 
#     depth: tuple(int,int)

class SnapshotDictSchema():
    """snapshot summary schema."""

    def __init__(self):
        self.bidPQ = {}
        self.bidPN = {}
        self.offerPQ = {}
        self.offerPN = {}
        
    @property
    def getBidPQ(self)->dict:
        return self.bidPQ
    @property
    def getBidPN(self)->dict:
        return self.bidPN
    @property
    def getOfferPQ(self)->dict:
        return self.offerPQ
    @property
    def getOfferPN(self)->dict:
        return self.offerPN
    
    def initSeries(self,snapshot:Series):
        bidOrderQtys = list(filter(lambda x: x is not None,ast.literal_eval(snapshot['BidOrderQty_S'].replace('null','None'))))

        bidPrice = list(filter(lambda x: x is not None,ast.literal_eval(snapshot['BidPrice_S'].replace('null','None'))))  
        bidNumOrders = list(filter(lambda x: x is not None,ast.literal_eval(snapshot['BidNumOrders_S'].replace('null','None'))))   


        offerOrderQtys = list(filter(lambda x: x is not None,ast.literal_eval(snapshot['OfferOrderQty_S'].replace('null','None'))))    
        offerPrice = list(filter(lambda x: x is not None,ast.literal_eval(snapshot['OfferPrice_S'].replace('null','None'))))   
        offerNumOrders = list(filter(lambda x: x is not None,ast.literal_eval(snapshot['OfferNumOrders_S'].replace('null','None'))))

        self.bidPQ = dict(zip(bidPrice,bidOrderQtys))
        self.bidPN = dict(zip(bidPrice,bidNumOrders))
        
        self.offerPQ = dict(zip(offerPrice,offerOrderQtys))
        self.offerPN = dict(zip(offerPrice,offerNumOrders))

    def initList(self,bidPrice:list,bidOrderQtys:list,bidNumOrders:list,
                       offerPrice:list,offerOrderQtys:list,offerNumOrders:list):


        self.bidPQ = dict(zip(bidPrice,bidOrderQtys))
        self.bidPN = dict(zip(bidPrice,bidNumOrders))
        
        self.offerPQ = dict(zip(offerPrice,offerOrderQtys))
        self.offerPN = dict(zip(offerPrice,offerNumOrders))     
        
    def compare_dict(self,dict_a, dict_b, show_value_diff=True):
      result = {}
      result['added']   = {k: dict_a[k] for k in set(dict_a) - set(dict_b)}
      result['removed'] = {k: dict_b[k] for k in set(dict_b) - set(dict_a)}
      if show_value_diff:
        common_keys =  set(dict_a) & set(dict_b)
        result['value_diffs'] = {
          k:(dict_a[k], dict_b[k])
          for k in common_keys
          if dict_a[k] != dict_b[k]
        }
      return result
  

    def __sub__(self,b_snapshot_dict)->dict:
        # 对于存在bid的，取最小，对于不存在的，最小bid就是0，要考虑到不存在的情况
        minBid = min(self.bidPQ.keys()) if self.bidPQ.keys() else 0
        # 对于存在offer的，取最大
        maxOffer = max(self.offerPQ.keys()) if self.bidPQ.keys() else float("inf")
        
 
        bidPQDif = self.compare_dict(self.bidPQ,b_snapshot_dict.getBidPQ)
        bidPNDif = self.compare_dict(self.bidPN,b_snapshot_dict.getBidPN)

        offerPQDif = self.compare_dict(self.offerPQ,b_snapshot_dict.getOfferPQ)
        offerPNDif = self.compare_dict(self.offerPN,b_snapshot_dict.getOfferPN)

        # 对于这些字典进行truncate处理，避免snapshot对比出现
        # 公布的snapshot范围以外的
        filteredBidPQDif = {}
        filteredBidPNDif = {}
        filteredOfferPQDif = {}
        filteredOfferPNDif = {}
        
        
        
        filteredBidPQDif['added'] = dict(filter(lambda item: float(item[0]) > minBid, bidPQDif['added'].items()))
        filteredBidPNDif['added'] = dict(filter(lambda item: float(item[0]) > minBid, bidPNDif['added'].items()))
        
        filteredOfferPQDif['added'] = dict(filter(lambda item: float(item[0]) < maxOffer , offerPQDif['added'].items()))
        filteredOfferPNDif['added'] = dict(filter(lambda item: float(item[0]) < maxOffer , offerPNDif['added'].items()))
                
        filteredBidPQDif['removed'] = dict(filter(lambda item: float(item[0]) > minBid, bidPQDif['removed'].items()))
        filteredBidPNDif['removed'] = dict(filter(lambda item: float(item[0]) > minBid, bidPNDif['removed'].items()))
        
        filteredOfferPQDif['removed'] = dict(filter(lambda item: float(item[0]) < maxOffer , offerPQDif['removed'].items()))
        filteredOfferPNDif['removed'] = dict(filter(lambda item: float(item[0]) < maxOffer , offerPNDif['removed'].items()))
                
        filteredBidPQDif['value_diffs'] = dict(filter(lambda item: float(item[0]) > minBid, bidPQDif['value_diffs'].items()))
        filteredBidPNDif['value_diffs'] = dict(filter(lambda item: float(item[0]) > minBid, bidPNDif['value_diffs'].items()))
        
        filteredOfferPQDif['value_diffs'] = dict(filter(lambda item: float(item[0]) < maxOffer , offerPQDif['value_diffs'].items()))
        filteredOfferPNDif['value_diffs'] = dict(filter(lambda item: float(item[0]) < maxOffer , offerPNDif['value_diffs'].items()))
                
        
        return({'BidPQ':filteredBidPQDif,'BidPN':filteredBidPNDif,
                'OfferPQ':filteredOfferPQDif,'OfferPN':filteredOfferPNDif})
# ss = SnapshotDictSchema()
# qq = SnapshotDictSchema()

# ss.initSeries(snapshot)
# qq.initSeries(snapshot)

# ss-qq
