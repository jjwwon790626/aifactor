# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 08:41:10 2025

@author: GF
"""
#%% 导入所需要的包
import datetime
import time
from pprint import pp
import pandas as pd
import copy
from order_matching.matching_engine import MatchingEngine
from order_matching.order import LimitOrder
from order_matching.order import MarketOrder
from order_matching.order import MarketLimitOrder, BestOfPartyOrder, FAK5LOrder, FAKAOrder, FOKAOrder
from order_matching.side import Side
from order_matching.orders import Orders
from order_matching.order_book import OrderBook
from order_matching.execution import Execution

from order_matching.schemas import SnapshotDictSchema


#%% 初始化一个订单簿，初始订单为6档
matching_engine = MatchingEngine(seed=123)

for i in range(1,7):
    side = Side.SELL
    price = 100 + i
    size = 100*i
    timestamp = datetime.datetime.now()
    order_id = i
    thisOrder = LimitOrder(side=side, price=price, size=size, timestamp=timestamp, order_id=int(order_id),trader_id=int(order_id))
    matching_engine.match(timestamp=timestamp,orders=Orders([thisOrder]))
    
# 查看初始化好了的订单簿
internalSnapshot = matching_engine.get_snapshot().summary()

#%%
#  首先测试marketLimitOrder，
matching_engine_marketlimitorder = copy.deepcopy(matching_engine)

side = Side.BUY
size = 500
timestamp = datetime.datetime.now()
order_id = 8
thisOrder = MarketLimitOrder(side=side, size=size, timestamp=timestamp, order_id=int(order_id), trader_id=int(order_id))
matching_engine_marketlimitorder.match(timestamp=timestamp,orders=Orders([thisOrder]))

internalSnapshotMarketlimitorder = matching_engine_marketlimitorder.get_snapshot().summary()

# 再试一次，看是否市价会根据其变化
side = Side.BUY
size = 500
timestamp = datetime.datetime.now()
order_id = 9
thisOrder = MarketLimitOrder(side=side, size=size, timestamp=timestamp, order_id=int(order_id), trader_id=int(order_id))
matching_engine_marketlimitorder.match(timestamp=timestamp,orders=Orders([thisOrder]))

internalSnapshotMarketlimitorder = matching_engine_marketlimitorder.get_snapshot().summary()

# 再试一个恰好完全成交的
side = Side.BUY
size = 300
timestamp = datetime.datetime.now()
order_id = 10
thisOrder = MarketLimitOrder(side=side, size=size, timestamp=timestamp, order_id=int(order_id), trader_id=int(order_id))
matching_engine_marketlimitorder.match(timestamp=timestamp,orders=Orders([thisOrder]))

internalSnapshotMarketlimitorder = matching_engine_marketlimitorder.get_snapshot().summary()


# 再测试一个不足的
side = Side.BUY
size = 100
timestamp = datetime.datetime.now()
order_id = 11
thisOrder = MarketLimitOrder(side=side, size=size, timestamp=timestamp, order_id=int(order_id), trader_id=int(order_id))
matching_engine_marketlimitorder.match(timestamp=timestamp,orders=Orders([thisOrder]))

internalSnapshotMarketlimitorder = matching_engine_marketlimitorder.get_snapshot().summary()

# 以上是对marketLimitOrder的测试，全部通过

#%% 接下来测试FAK5L
matching_engine_fak5l = copy.deepcopy(matching_engine)

# 先测试只能打完1档位的
side = Side.BUY
size = 100
timestamp = datetime.datetime.now()
order_id = 8
thisOrder = FAK5LOrder(side=side,size=size, timestamp=timestamp, order_id=int(order_id), trader_id=int(order_id))
matching_engine_fak5l.match(timestamp=timestamp,orders=Orders([thisOrder]))

internalSnapshotFAK5L =matching_engine_fak5l.get_snapshot().summary()
                                          

# 然后测试能打完5档位的
side = Side.BUY
size = 2100
timestamp = datetime.datetime.now()
order_id = 9
thisOrder = FAK5LOrder(side=side,size=size, timestamp=timestamp, order_id=int(order_id), trader_id=int(order_id))
matching_engine_fak5l.match(timestamp=timestamp,orders=Orders([thisOrder]))

internalSnapshotFAK5L =matching_engine_fak5l.get_snapshot().summary()
                                          
# 再测试只能打完1档多的
matching_engine_fak5l = copy.deepcopy(matching_engine)

# 先测试只能打完1档位的
side = Side.BUY
size = 200
timestamp = datetime.datetime.now()
order_id = 8
thisOrder = FAK5LOrder(side=side,size=size, timestamp=timestamp, order_id=int(order_id), trader_id=int(order_id))
matching_engine_fak5l.match(timestamp=timestamp,orders=Orders([thisOrder]))

internalSnapshotFAK5L = matching_engine_fak5l.get_snapshot().summary()

# 测试完成，无任何问题

#%% 接下来测试FAKA订单
matching_engine_faka = copy.deepcopy(matching_engine)

# 先测试只能打完1档位的
side = Side.BUY
size = 100
timestamp = datetime.datetime.now()
order_id = 8
thisOrder = FAKAOrder(side=side,size=size, timestamp=timestamp, order_id=int(order_id), trader_id=int(order_id))
matching_engine_faka.match(timestamp=timestamp,orders=Orders([thisOrder]))

internalSnapshotFAKA = matching_engine_faka.get_snapshot().summary()



matching_engine_faka = copy.deepcopy(matching_engine)

# 再测试还剩一档位的
side = Side.BUY
size = 2000
timestamp = datetime.datetime.now()
order_id = 9
thisOrder = FAKAOrder(side=side,size=size, timestamp=timestamp, order_id=int(order_id), trader_id=int(order_id))
matching_engine_faka.match(timestamp=timestamp,orders=Orders([thisOrder]))

internalSnapshotFAKA = matching_engine_faka.get_snapshot().summary()



# 最后测试还剩一档位不剩的
side = Side.BUY
size = 2000
timestamp = datetime.datetime.now()
order_id = 10
thisOrder = FAKAOrder(side=side,size=size, timestamp=timestamp, order_id=int(order_id), trader_id=int(order_id))
matching_engine_faka.match(timestamp=timestamp,orders=Orders([thisOrder]))

internalSnapshotFAKA = matching_engine_faka.get_snapshot().summary()



#%% 接下来测试FOKA订单
matching_engine_foka = copy.deepcopy(matching_engine)

# 测试全部kill的情况
side = Side.BUY
size = 2700
timestamp = datetime.datetime.now()
order_id = 10
thisOrder = FOKAOrder(side=side,size=size, timestamp=timestamp, order_id=int(order_id), trader_id=int(order_id))
trade_foka = matching_engine_foka.match(timestamp=timestamp,orders=Orders([thisOrder]))

internalSnapshotFOKA = matching_engine_foka.get_snapshot().summary()


# 测试成交1档的情况
side = Side.BUY
size = 100
timestamp = datetime.datetime.now()
order_id = 11
thisOrder = FOKAOrder(side=side,size=size, timestamp=timestamp, order_id=int(order_id), trader_id=int(order_id))
matching_engine_foka.match(timestamp=timestamp,orders=Orders([thisOrder]))

internalSnapshotFOKA = matching_engine_foka.get_snapshot().summary()


# 测试成交2档位的情况
side = Side.BUY
size = 500
timestamp = datetime.datetime.now()
order_id = 12
thisOrder = FOKAOrder(side=side,size=size, timestamp=timestamp, order_id=int(order_id), trader_id=int(order_id))
matching_engine_foka.match(timestamp=timestamp,orders=Orders([thisOrder]))

internalSnapshotFOKA = matching_engine_foka.get_snapshot().summary()

# 测试全部成交的情况
side = Side.BUY
size = 1500
timestamp = datetime.datetime.now()
order_id = 12
thisOrder = FOKAOrder(side=side,size=size, timestamp=timestamp, order_id=int(order_id), trader_id=int(order_id))
matching_engine_foka.match(timestamp=timestamp,orders=Orders([thisOrder]))

internalSnapshotFOKA = matching_engine_foka.get_snapshot().summary()
# 测试完成，结果与预期一致


#%% 最后测试本方最优
matching_engine_bestofpartyorder = copy.deepcopy(matching_engine)

# 测试全部kill的情况
side = Side.BUY
size = 2700
timestamp = datetime.datetime.now()
order_id = 10
thisOrder = BestOfPartyOrder(side=side,size=size, timestamp=timestamp, order_id=int(order_id), trader_id=int(order_id))
matching_engine_bestofpartyorder.match(timestamp=timestamp,orders=Orders([thisOrder]))

internalSnapshotBestofpartyorder = matching_engine_bestofpartyorder.get_snapshot().summary()

#测试反过来的情况
side = Side.SELL
size = 2700
timestamp = datetime.datetime.now()
order_id = 10
thisOrder = BestOfPartyOrder(side=side,size=size, timestamp=timestamp, order_id=int(order_id), trader_id=int(order_id))
matching_engine_bestofpartyorder.match(timestamp=timestamp,orders=Orders([thisOrder]))

internalSnapshotBestofpartyorder = matching_engine_bestofpartyorder.get_snapshot().summary()

# 再测试一下加上去的订单能不能被成交

side = Side.BUY
size = 300
timestamp = datetime.datetime.now()
order_id = 12
thisOrder = FOKAOrder(side=side,size=size, timestamp=timestamp, order_id=int(order_id), trader_id=int(order_id))
trade_matching_engine_bestofpartyorder = matching_engine_bestofpartyorder.match(timestamp=timestamp,orders=Orders([thisOrder]))


side = Side.BUY
size = 300
timestamp = datetime.datetime.now()
order_id = 13
thisOrder = FOKAOrder(side=side,size=size, timestamp=timestamp, order_id=int(order_id), trader_id=int(order_id))
trade_matching_engine_bestofpartyorder = matching_engine_bestofpartyorder.match(timestamp=timestamp,orders=Orders([thisOrder]))