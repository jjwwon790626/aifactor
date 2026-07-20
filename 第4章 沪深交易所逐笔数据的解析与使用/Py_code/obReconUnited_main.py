# -*- coding: utf-8 -*-
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
from order_matching.executed_trades import ExecutedTrades
from order_matching.trade import Trade

# 案例2
code = '000725'
date = '20250407'

match code[0]:
    case '0':
        issueType = 'szstock'
    case '1':
        issueType = 'szfund'
    case '3':
        issueType = 'szcybstock'
    case '5':
        issueType = 'shfund'
    case '6':
        if code[:3] == '688':
            issueType = 'shkcbstock'
        else:
            issueType = 'shstock'
            
# 完成归类以后，方便进行下一步
#%% 各个品种其时间段是不一样的
# 首先定义各个交易时间常数，对于任何一个品种，其
if issueType in ['shfund','shkcbstock','shstock']:
    callAuctionTime = datetime.datetime(int(date[:4]),int(date[4:6]),int(date[6:8]),9,25,1)
if issueType in ['szstock','szfund','szcybstock']:
    callAuctionTime = datetime.datetime(int(date[:4]),int(date[4:6]),int(date[6:8]),9,25,00)
    # 对于深交所的品种存在这个参数：是否通过成交来复现市价单
    RECONSTRUCT_THROUGH_TRADE = True
    
# 如果是上交所基金，不存在收盘集合竞价，如果是股票，就存在收盘集合竞价
if issueType in ['shfund']:
    pass
if issueType in ['shkcbstock','shstock']:
    closeCallAuctionTime = datetime.datetime(int(date[:4]),int(date[4:6]),int(date[6:8]),14,57,00)
if issueType in ['szstock','szfund','szcybstock']:
    closeCallAuctionTime = datetime.datetime(int(date[:4]),int(date[4:6]),int(date[6:8]),14,57,00)
    
if issueType in ['shfund','shkcbstock','shstock']:
# 无论是股票还是基金，都是3点收盘，延长1秒是为了使得可能略微的延迟被纳入考虑，不延迟太多是为了避免纳入盘后大宗交易
    closeTime = datetime.datetime(int(date[:4]),int(date[4:6]),int(date[6:8]),15,00,1)
if issueType in ['szstock','szfund','szcybstock']:
    closeTime = datetime.datetime(int(date[:4]),int(date[4:6]),int(date[6:8]),15,00,00)

if issueType in ['shfund','shkcbstock','shstock']:
# 这些列是entrust,trade,snapshot公共的列
    SKIP_COLUMNS = ['TradeTime','LocalTime', 'BizIndex','ApplSeqNum','OfferApplSeqNum','BidApplSeqNum']
if issueType in ['szstock','szfund','szcybstock']:
    SKIP_COLUMNS = ['TradeTime','LocalTime','ApplSeqNum','SeqNo','BidApplSeqNum','OfferApplSeqNum']
    


#%% 导入csv
entrust = pd.read_csv('E:\\level2\\{0}\\entrust\\entrust_{0}_{1}.csv'.format(code,date))
tradeTable = pd.read_csv('E:\\level2\\{0}\\trade\\trade_{0}_{1}.csv'.format(code,date))
snapshot = pd.read_csv('E:\\level2\\{0}\\snapshot\\snapshot_{0}_{1}.csv'.format(code,date))

if issueType in ['shfund','shkcbstock','shstock']:
# 这些列是entrust,trade,snapshot公共的列
    entrust['ApplSeqNum'] = entrust['ApplSeqNum'].astype('Int64')
    entrust['BizIndex'] = entrust['BizIndex'].astype('Int64')
    tradeTable['BizIndex'] = tradeTable['BizIndex'].astype('Int64')
    tradeTable['OfferApplSeqNum'] = tradeTable['OfferApplSeqNum'].astype('Int64')
    tradeTable['BidApplSeqNum'] = tradeTable['BidApplSeqNum'].astype('Int64')
if issueType in ['szstock','szfund','szcybstock']:
    entrust['ApplSeqNum'] = entrust['ApplSeqNum'].astype('Int64')
    tradeTable['OfferApplSeqNum'] = tradeTable['OfferApplSeqNum'].astype('Int64')
    tradeTable['BidApplSeqNum'] = tradeTable['BidApplSeqNum'].astype('Int64')

#%%# 读取了entrust以后，来获取entrust
# 去除包含Call 和Trade 这两种标识集合竞价与连续竞价开始的标识符
entrust_effective = entrust[entrust['Side'].isin(['B','S'])]
entrust_effective.loc[:,'TradeTime'] = entrust_effective['TradeTime'].apply(lambda x:datetime.datetime.strptime(str(x),"%Y.%m.%dT%H:%M:%S.%f"))
entrust_effective = entrust_effective.rename(columns={col:col+"_E" for col in entrust_effective.columns if col not in SKIP_COLUMNS})
entrust_effective.loc[:,'type'] = 'entrust'

tradeTable.loc[:,'TradeTime'] = tradeTable['TradeTime'].apply(lambda x:datetime.datetime.strptime(str(x),"%Y.%m.%dT%H:%M:%S.%f"))
tradeTable = tradeTable.rename(columns={col:col+"_T" for col in tradeTable.columns if col not in SKIP_COLUMNS})
# 这个type必须在其后，否则会命名有问题带上后缀
tradeTable.loc[:,'type'] = 'trade'

# 去除包含Call 和Trade 这两种标识集合竞价与连续竞价开始的标识符
snapshot.loc[:,'TradeTime'] = snapshot['TradeTime'].apply(lambda x:datetime.datetime.strptime(str(x),"%Y.%m.%dT%H:%M:%S.%f"))
snapshot = snapshot.rename(columns={col:col+"_S" for col in snapshot.columns if col not in SKIP_COLUMNS})
# 这个type也必须在其后
snapshot.loc[:,'type'] = 'snapshot'

aggregateTable = pd.concat([entrust_effective,tradeTable,snapshot])
aggregateTable.sort_values('TradeTime',inplace=True)
aggregateTable.reset_index(drop=True,inplace=True)

BUY_SELL_DICT = {'B':Side.BUY,'S':Side.SELL}

# 同时初始化一下本方的trade表用来记录和与公布的trade进行比对
selfExecutedTrades = ExecutedTrades()
# 并且初始化一下官方的trade表
officialExecutedTrades = ExecutedTrades()

# 这个便于观察新到达的委托或者成交是否存在，必须初始化这个变量
# 这个变量进存在于上交所的撮合
if issueType in ['shfund','shkcbstock','shstock']:
    applSeqNumList = []
# 深交所的品种不需要这个变量
#%% 为了进行开盘集合竞价，进行初始化
# 初始化撮合引擎，随机数种子随机设置
matching_engine = MatchingEngine(seed=123)
callAuctionOrders = Orders()
callAuctionAggregateTable = aggregateTable[aggregateTable['TradeTime']<=callAuctionTime]

# 上交所同时根据TradeTime和BizIndex进行排序，BizIndex同时出现在entrust和trade
if issueType in ['shfund','shkcbstock','shstock']:
    callAuctionAggregateTable.sort_values(by=['TradeTime','BizIndex'],inplace=True)
if issueType in ['szstock','szfund','szcybstock']:
    callAuctionAggregateTable.sort_values(by=['TradeTime','ApplSeqNum'],inplace=True)
    
if issueType in ['shfund','shkcbstock','shstock']:
    for ind,row in callAuctionAggregateTable.iloc[:	].iterrows():
        match row['type']:
            case 'entrust':
                if row['OrderType_E']=='A':
                    side = BUY_SELL_DICT[row['Side_E']]
                    price = row['Price_E']
                    size = row['OrderQty_E']
                    timestamp = row['TradeTime']
                    order_id = row['ApplSeqNum']
                    # 这里要尤其当心传入的order_id的格式，因为LimitOrder初始化的时候会将传入的order_id直接转化为str类型，因此要先取int再传入，否则会出现'1.0'这样的order_id
                    callAuctionOrder = LimitOrder(side=side, price=price, size=size, timestamp=timestamp, order_id=int(order_id),trader_id=int(order_id))
                    callAuctionOrders.add([callAuctionOrder])
    
                # 这一段是处理上交所撤单逻辑，上交所的撤单在委托表即可查到，与深交所不同，其OrderType字段表示的是挂单还是撤单，而非市价限价单之区别
                if row['OrderType_E']=='D':
                    cancelApplySeqNum = int(row['ApplSeqNum'])
                    timestamp = row['TradeTime']
                    # 这里要尤其当心传入的order_id的格式，cancel_order函数会做一次转化，所以要确保取消的订单其格式一致，否则会出现取消问题
                    callAuctionOrders.cancel_order(cancelApplySeqNum)
            case 'trade':
                tradeQty = row['TradeQty_T']
                bidApplSeqNum = row['BidApplSeqNum']
                offerApplSeqNum = row['OfferApplSeqNum']
                # trade是一个自己写的函数，用于将集合竞价已经成交的部分订单扣减掉
                callAuctionOrders.trade(int(bidApplSeqNum), tradeQty)
                callAuctionOrders.trade(int(offerApplSeqNum), tradeQty)
    
if issueType in ['szstock','szfund','szcybstock']:
    for ind,row in callAuctionAggregateTable.iloc[:	].iterrows():
        match row['type']:
            case 'entrust':
                side = BUY_SELL_DICT[row['Side_E']]
                price = row['Price_E']
                size = row['OrderQty_E']
                timestamp = row['TradeTime']
                order_id = row['ApplSeqNum']
                # 这里要尤其当心传入的order_id的格式，因为LimitOrder初始化的时候会将传入的order_id直接转化为str类型，因此要先取int再传入，否则会出现'1.0'这样的order_id
                thisOrder = LimitOrder(side=side, price=price, size=size, timestamp=timestamp, order_id=int(order_id),trader_id=int(order_id))
                callAuctionOrders.add([thisOrder])

            case 'trade':
                if row['ExecType_T'] == 52 or row['ExecType_T'] == '52':
                    # ExecType是52代表在撤单
                    cancelApplySeqNum = int(max(row['BidApplSeqNum'],row['OfferApplSeqNum']))
                    timestamp = row['TradeTime']
                    callAuctionOrders.cancel_order(cancelApplySeqNum)
                elif row['ExecType_T'] == 70 or row['ExecType_T'] == '70':
                    # ExecType是70代表在成交
                    tradeQty = row['TradeQty_T']
                    bidApplSeqNum = row['BidApplSeqNum']
                    offerApplSeqNum = row['OfferApplSeqNum']
                    # trade是一个自己写的函数，用于将集合竞价已经成交的部分订单扣减掉
                    callAuctionOrders.trade(int(bidApplSeqNum), tradeQty)
                    callAuctionOrders.trade(int(offerApplSeqNum), tradeQty)
                    
# 集合竞价时间到，开始match
callAuctionTrades = matching_engine.match(timestamp=callAuctionTime,orders=callAuctionOrders)
# 然后获取集合竞价后的所有订单
afterCallAuctionAggregateTable = aggregateTable[aggregateTable['TradeTime']>callAuctionTime]                
# 查看集合竞价后的snapshot
afterCallAuctionSnapshot = matching_engine.snapshot.summary()

#%% 比较集合竞价后本方撮合与交易所提供的snapshot的部分
selfSnapshotDict = matching_engine.get_snapshot().snapshotDict    
officialSnapshotDict = SnapshotDictSchema()

if issueType in ['shfund','shkcbstock','shstock']:
    officialCallAuctionRow = aggregateTable[(aggregateTable['TradeTime']==callAuctionTime) & (aggregateTable['type']=='snapshot')]
if issueType in ['szstock','szfund','szcybstock']:
    officialCallAuctionRow = aggregateTable[(aggregateTable['TradingPhaseCode_S']=='B0')].iloc[0:1,:]
    
officialSnapshotDict.initSeries(officialCallAuctionRow.iloc[0])
dif = (officialSnapshotDict - selfSnapshotDict)

#%% 处理连续竞价
if issueType in ['shfund','shkcbstock','shstock']:
    afterCallAuctionAggregateTable.sort_values(by=['TradeTime','BizIndex'],inplace=True)
    for ind,row in afterCallAuctionAggregateTable.iloc[:].iterrows():
        # 默认是不需要进行匹配的信息，例如snapshot，例如Cancel
        matchIteration = False
        timestamp = row['TradeTime']
        # if timestamp>datetime.datetime(int(date[:4]),int(date[4:6]),int(date[6:8]),11,30,00):
        #    break    

    #%% 由于上交所会出现trade比entrust先到的情况，或者是trade到但是从来没有entrust的情况，因此需要先行还原
    #%% 上交所的处理部分                    
    # 上交所的从未在委托表里出现的成交需要从成交表里按照BidApplSeqNum 和 OfferApplSeqNum分别还原为买委托和卖委托，数量需要加总
    # 上交所的委托如果立即成交了一部分还需要将TradedQty加还到OrderQty里面
    # 否则无法还原为当时的委托
    # 对于上交所的品种，需要将没出现在entrustIDList里面的买委托，还原到timeWindow里面的买订单，只有
    # 订单量需要是汇总量，其他的委托只要是出现在委托表里面的，就直接读取委托表即可
    # 没出现过得订单才需要append上去entrustIDList

        match row['type']:
            case 'entrust':
                # 对于没出现在tradeList和ApplSeqNumList里面的，需要新增委托
                if row['OrderType_E']=='A':
                    order_id = row['ApplSeqNum']
                    # 对于凡是没出现过的订单，先从成交的部分加回，然后检查是否有成交过，成交的也要加回
                    if not (order_id in applSeqNumList or str(order_id) in applSeqNumList) :
                        # 先发现委托的成交直接加回来即可，因为后面的成交肯定是已经躺在订单簿里面的book_order的历史
                        row['OrderQty_E'] = row['OrderQty_E'] + row['TradedQty_E']
                        side = BUY_SELL_DICT[row['Side_E']]
                        price = row['Price_E']
                        size = row['OrderQty_E']
                        timestamp = row['TradeTime']
                        # 这里的order_id任何格式都可以，本来就是从数据表中直接取的
                        applSeqNumList.append(order_id)
                        thisOrder = LimitOrder(side=side, price=price, size=size, timestamp=timestamp, order_id=int(order_id),trader_id=int(order_id))
                        # 对于新的委托就直接加入列表并且成交
                        matchIteration = True
                    else:
                        pass
                if row['OrderType_E']=='D':
                    cancelApplySeqNum = int(row['ApplSeqNum'])
                    timestamp = row['TradeTime']
                    matching_engine.cancel_order(int(cancelApplySeqNum))
            case 'trade':
                # 对于成交的订单，直接全部加回
                # 因为直接发生了成交，所以后续的残余在订单簿的委托可能存在，
                # 也可能不存在，还那么就需要将后期的相同订单的成交（如果有）和委托（如果有）加回，还原成为
                # 原始的订单

                # 在处理的时候还是使用原始的格式合适
                # 不考虑成交的买卖两笔订单都在订单簿上没出现过的情况，因为那样意味着在事件顺序正确的情况下
                # 如果两笔都没出现，那么其委托顺序叫难以判断先后，也不便于累积订单来撮合
                # 委托号为较大的那个属于主动单
                incoming_order_id = int(max(row['BidApplSeqNum'],row['OfferApplSeqNum']))

                # 对于已经处理过了的订单，就不用处理了，这里只需要处理那些没有处理过的订单
                if not (incoming_order_id in applSeqNumList or str(incoming_order_id) in applSeqNumList):
                    # 如果在委托表里面能够找到这笔委托，直接从委托表还原
                    specificEntrustTable = afterCallAuctionAggregateTable[(afterCallAuctionAggregateTable['type']=='entrust') & (afterCallAuctionAggregateTable['ApplSeqNum']==incoming_order_id) & (afterCallAuctionAggregateTable['OrderType_E']=='A')]
                    if len(specificEntrustTable)>0:
                        # 找到的是这笔订单的委托号
                        order_id = incoming_order_id
                        specificEntrustTable.loc[:,'OrderQty_E'] = specificEntrustTable['OrderQty_E'] + specificEntrustTable['TradedQty_E']
                        # 提取委托所在行
                        specificRow = specificEntrustTable.iloc[0]
                        # 下面参考委托的处理逻辑         
                        side = BUY_SELL_DICT[specificRow['Side_E']]
                        price = specificRow['Price_E']                  
                        size = specificRow['OrderQty_E']
                        execution = Execution.LIMIT
                        timestamp = specificRow['TradeTime']
                    # 如果是一笔纯打单且完全成交，那么就不会在entrust表上留下任何足迹
                    else:
                        # 对于纯成交的打单，主动方向就是这笔订单的方向
                        # 只有那些需要纯成交的才需要通过加总还原
                        order_id = incoming_order_id
                        side = Side.BUY if row['TradeBSFlag_T'] == 'B' else Side.SELL
                        # 上交所的成交表里面不区分市价单和限价单，一律按照限价单处理
                        execution = Execution.LIMIT
                        # 一瞬间引起的一个或者多个成交，可以认为是在同一个时间戳
                        timestamp = row['TradeTime']
                        # 需要修改的就是数量，所以找出所有相应的订单
                        if side == Side.BUY:
                            # 买单就查看所有BidApplSeqNum相等的trade
                            specificTradeTable = afterCallAuctionAggregateTable[(afterCallAuctionAggregateTable['type']=='trade') & (afterCallAuctionAggregateTable['BidApplSeqNum']==incoming_order_id)]
                            # 可能会打到其他的订单，因此必须确保买单的price是最高的，而卖单的price是最低的
                            price = max(specificTradeTable['TradePrice_T'])
                        else:
                            specificTradeTable = afterCallAuctionAggregateTable[(afterCallAuctionAggregateTable['type']=='trade') & (afterCallAuctionAggregateTable['OfferApplSeqNum']==incoming_order_id)]
                            price = min(specificTradeTable['TradePrice_T'])
                        size = specificTradeTable['TradeQty_T'].sum()
                    applSeqNumList.append(order_id)
                    thisOrder = LimitOrder(side=side, price=price, size=size, timestamp=timestamp, order_id=int(order_id),trader_id=int(order_id))
                    matchIteration = True
                    # 还原出来的订单需要处理
                    
                # 加入成交对比的逻辑   
                # 下面的代码用来生成一个trade对象从而存储起来，便于和我们的撮合引擎撮合出来的trade进行对比
                thisTrade = Trade(
                    side = Side.BUY if row['TradeBSFlag_T'] == 'B' else Side.SELL,
                    price = row['TradePrice_T'],
                    size = row['TradeQty_T'],
                    incoming_order_id = str(int(max(row['BidApplSeqNum'],row['OfferApplSeqNum']))),
                    book_order_id = str(int(min(row['BidApplSeqNum'],row['OfferApplSeqNum']))),
                    execution = Execution.LIMIT,
                    trade_id = None,
                    timestamp = row['TradeTime'])
                officialExecutedTrades += ExecutedTrades([thisTrade])
     
        # 如果matchIteration为真，就要进行撮合
        if matchIteration == True:
            iterationTrade = matching_engine.match(timestamp=timestamp,orders=Orders([thisOrder]))
            iterationTrade.to_frame()
            selfExecutedTrades += iterationTrade
            internalSnapshot = matching_engine.get_snapshot().summary()
      
        if row['type']=='snapshot':
            # 对于快照行情
            selfSnapshotDict = matching_engine.get_snapshot().snapshotDict
            officialSnapshotDict = SnapshotDictSchema()
            officialSnapshotDict.initSeries(row)
            dif = officialSnapshotDict - selfSnapshotDict
            
if issueType in ['szstock','szfund','szcybstock']:
    #%% 开始处理连续竞价部分
    afterCallAuctionAggregateTable.sort_values(by=['TradeTime','ApplSeqNum'],inplace=True)

    for ind,row in afterCallAuctionAggregateTable.iloc[:].iterrows():
        print(ind)
    #     # 默认是不需要进行匹配的信息，例如snapshot，例如Cancel
        matchIteration = False
    # #    if timestamp>datetime.datetime(int(date[:4]),int(date[4:6]),int(date[6:8]),11,30,00):
    # #        break
    #     # 测试用
    #     #row = afterCallAuctionAggregateTable.iloc[5]
        timestamp = row['TradeTime']	

        
        #对于深交所的品种
        match row['type']:
            case 'entrust':
                orderType = row['OrderType_E']
                # 这意味着是限价单
                if orderType == 2 or orderType=='2':
                    side = BUY_SELL_DICT[row['Side_E']]
                    price = row['Price_E']
                    size = row['OrderQty_E']
                    timestamp = row['TradeTime']
                    order_id = row['ApplSeqNum']
                    thisOrder = LimitOrder(side=side, price=price, size=size, timestamp=timestamp, order_id=int(order_id),trader_id=int(order_id))
                    matchIteration = True
                if orderType == 1 or orderType=='1':
                    # 第一种逻辑是根据委托来复现                
                    if RECONSTRUCT_THROUGH_TRADE == False:
                        side = BUY_SELL_DICT[row['Side_E']]
                        size = row['OrderQty_E']
                        timestamp = row['TradeTime']
                        order_id = row['ApplSeqNum']
                        thisOrder = MarketOrder(side=side, size=size, timestamp=timestamp, order_id=int(order_id),trader_id=int(order_id))
                        matchIteration = True
                    # 另一条逻辑是，一旦发现这种市价单，立即去trade表里面寻找这笔订单对应的订单号，然后根据查询出来的成交和撤单，
                    # 倒推出原始订单的委托到底是哪一笔，不能区分的主要是3种订单，FAK5LOrder, FAKAOrder, FOKAOrder
                    else:
                        side = BUY_SELL_DICT[row['Side_E']]
                        size = row['OrderQty_E']
                        timestamp = row['TradeTime']
                        order_id = row['ApplSeqNum']
                        # 唯一需要添加 int是因为需要对比，凡是和dataframe对比的都是int
                        mktOrderApplSeqNum = int(row['ApplSeqNum'])
                        thisOrderTrade = afterCallAuctionAggregateTable[((afterCallAuctionAggregateTable['type']=='trade') & ((afterCallAuctionAggregateTable['BidApplSeqNum']==mktOrderApplSeqNum) | (afterCallAuctionAggregateTable['OfferApplSeqNum']==mktOrderApplSeqNum)))]
                        # 如果是市价的打对手盘总共有几种情况，注意，这里只能根据成交倒推，但实际上原始订单不一定是这种类型，只是起到了相同的成交和撤单效果
                        #（一）对手方最优价格申报；最终只会有一种价格，且只有成交  只有一种价格
                        #（二）本方最优价格申报；
                        #（三）最优五档即时成交剩余撤销申报；   最终会有多种价格且会撤单  有小于等于5种价格且有撤单
                        #（四）即时成交剩余撤销申报；   只有一种价格，且会有撤销  有多于5种价格 和撤销
                        #（五）全额成交或撤销申报；  只有一个撤销单  ，或者是大于5个价格的成交单
                        # 可以根据盘口来测算，如果是对手盘最优价格申报，就转化为一个一开始设置的定量的限价单，如果不存在就取消
                        # 最优五档则是需要读取对手盘最优五档盘口的挂单量，如果0档就直接撤销，一直往上数所有的非空档位，然后发送数量和价格，价格取第五档或者在不满五档时取最大
                        # 下的订单的价格是五档以外取五档，五档以内取买单就取卖价最高，卖单就取买价最低
                        # 直到成交为止那么需要确定的是这五档中哪一档为需要发送的价格，只需要确定价格，还需要确定min(直到这一档的总量，和订单的总量)即可
                        # 即时成交剩余撤销，取的是第一档盘口，数量则是发送的数量，剩余的是一个撤销订单，可以把这两个
                        # 把前五档的数量加起来，和订单数量比较，如果没有5档，就把有的数量加起来，如果大于等于那个
                        # 市价单数量，就发一个对应的价格的限价单；如果小于那个市价单数量，数量就取能取到的限价订单簿的数量，然后剩余订单直接撤销
                        # 下面是获取不同档位的成交的档数
                        thisOrderTradeTrade = thisOrderTrade[thisOrderTrade['ExecType_T']==70]
                        tradePriceLevel = len(thisOrderTradeTrade['TradePrice_T'].unique())
                        # 下面是获取的取消的档数，其实应该只有0个和1个
                        thisOrderTradeCancel = thisOrderTrade[thisOrderTrade['ExecType_T']==52]
                        cancelLevel = len(thisOrderTradeCancel)
                        
                        # 只成交在一个价格档位默认是MarketLimitOrder，实际上也有可能是数量不足的FAK5Lorder或者是FAKAOrder或者是FOKAOrder
                        if tradePriceLevel == 1 and cancelLevel == 0:
                            thisOrder = MarketLimitOrder(side=side, size=size, timestamp=timestamp, order_id=int(order_id), trader_id=int(order_id))

                        # 成交在5个档位以内且有撤销就满足FAK5LOrder，但也有可能是FAKAOrder,如果没有撤销也有可能是FOKAOrder
                        elif 5 >= tradePriceLevel >= 1 and cancelLevel >= 0:
                            if cancelLevel == 0:
                                thisOrder = FAK5LOrder(side=side, size=size, timestamp=timestamp,
                                                       order_id=int(order_id), trader_id=int(order_id))
                            else:
                                cancelTime = thisOrderTradeCancel.iloc[0]['TradeTime']
                                if cancelTime == timestamp:
                                    thisOrder = FAK5LOrder(side=side, size=size, timestamp=timestamp,
                                                           order_id=int(order_id), trader_id=int(order_id))
                                else:
                                    # 如果后续发生撤单，说明委托的时候没有撤单
                                    thisOrder = MarketLimitOrder(side=side, size=size, timestamp=timestamp,
                                                                 order_id=int(order_id), trader_id=int(order_id))
                             
                        # 成交在大于5的多个档位就满足FAKAOrder，如果没有撤销也有可能是FOKAOrder
                        if tradePriceLevel > 5 and cancelLevel >= 0:
                            thisOrder = FAKAOrder(side=side, size=size, timestamp=timestamp, order_id=int(order_id), trader_id=int(order_id))
                               
                        # 只有撤单的市价单就满足FOKAOrder                               
                        if tradePriceLevel ==0 and cancelLevel > 0: # FOKA就会只有撤单没有成交
                            thisOrder = FOKAOrder(side=side, size=size, timestamp=timestamp, order_id=int(order_id), trader_id=int(order_id))
                                                   
                        # 对于这些市价单也需要
                        matchIteration = True
                if orderType == 'U':
                    # 本方最优价格申报，BestOfPartyOrder
                    side = BUY_SELL_DICT[row['Side_E']]
                    size = row['OrderQty_E']
                    timestamp = row['TradeTime']
                    order_id = int(row['ApplSeqNum'])
                    thisOrder = BestOfPartyOrder(side=side, size=size, timestamp=timestamp, order_id=int(order_id),trader_id=int(order_id))
                    matchIteration = True


            case 'trade':
                # ExecType是 52 代表在撤单
                if row['ExecType_T'] == 52 or row['ExecType_T'] == '52' :
                    # 必须导入int类型的，否则就匹配不到了
                    cancelApplySeqNum = int(max(row['BidApplSeqNum'],row['OfferApplSeqNum']))
                    timestamp = row['TradeTime']
                    matching_engine.cancel_order(cancelApplySeqNum)
                elif row['ExecType_T'] == 70 or row['ExecType_T'] == '70':
                    # 加入成交对比的逻辑
                    thisTrade = Trade(
                        side = Side.BUY if row['TradeBSFlag_T'] == 'B' else Side.SELL,
                        price = row['TradePrice_T'],
                        size = row['TradeQty_T'],
                        incoming_order_id = str(int(max(row['BidApplSeqNum'],row['OfferApplSeqNum']))),
                        book_order_id = str(int(min(row['BidApplSeqNum'],row['OfferApplSeqNum']))),
                        execution = Execution.LIMIT,
                        trade_id = None,
                        timestamp = row['TradeTime'])
                    officialExecutedTrades += ExecutedTrades([thisTrade])
     
        if matchIteration == True:
            # 撮合并输出

            iterationTrade = matching_engine.match(timestamp=timestamp,orders=Orders([thisOrder]))
            iterationTrade.to_frame()
            selfExecutedTrades += iterationTrade
            internalSnapshot = matching_engine.get_snapshot().summary()
            
        if row['type']=='snapshot':
            # 对于快照行情
            selfSnapshotDict = matching_engine.get_snapshot().snapshotDict    
            officialSnapshotDict = SnapshotDictSchema()
            officialSnapshotDict.initSeries(row)
            dif = officialSnapshotDict - selfSnapshotDict

# 除了上交所基金以外其他都有收盘集合竞价
if issueType in ['szstock','szfund','szcybstock']:

    #%% 处理收盘集合竞价部分
    # 找出其中与开盘集合竞价相关的订单和trade以及snapshot
    # 先提取收盘集合竞价之前未成交的订单
    closeCallAuctionOrders = matching_engine.get_all_orders()
    # 重新初始化一个收盘集合竞价撮合引擎
    close_matching_engine = MatchingEngine(seed=456)
    closeCallAuctionAggregateTable = aggregateTable[(aggregateTable['TradeTime']>=closeCallAuctionTime) & ((aggregateTable['TradeTime']<=closeTime))]
    closeCallAuctionAggregateTable.sort_values(by=['TradeTime','ApplSeqNum'],inplace=True)
    for ind,row in closeCallAuctionAggregateTable.iloc[:].iterrows():
        timestamp = row['TradeTime']
        match row['type']:
            case 'entrust':
                side = BUY_SELL_DICT[row['Side_E']]
                price = row['Price_E']
                size = row['OrderQty_E']
                timestamp = row['TradeTime']
                order_id = row['ApplSeqNum']
                thisOrder = LimitOrder(side=side, price=price, size=size, timestamp=timestamp, order_id=int(order_id),trader_id=int(order_id))
                closeCallAuctionOrders.add([thisOrder])
    
            case 'trade':
                # 收盘集合竞价不允许撤单
                # if row['ExecType_T'] == 52 or row['ExecType_T'] == '52':
                #     # ExecType是 52 代表在撤单
                #     cancelApplySeqNum = int(max(row['BidApplSeqNum_T'],row['OfferApplSeqNum_T']))
                #     timestamp = row['TradeTime']
                #     closeCallAuctionOrders.cancel_order(cancelApplySeqNum)
                tradeQty = row['TradeQty_T']
                bidApplSeqNum = row['BidApplSeqNum']
                offerApplSeqNum = row['OfferApplSeqNum']
                closeCallAuctionOrders.trade(int(bidApplSeqNum), tradeQty)
                closeCallAuctionOrders.trade(int(offerApplSeqNum), tradeQty)
    
    closeCallAuctionTrades = close_matching_engine.match(timestamp=closeCallAuctionTime,orders=closeCallAuctionOrders)
    afterCallAuctionSnapshot = close_matching_engine.snapshot.summary()
    # 收盘snapshot的时间取决于交易所发布的时间，可能需要手工调整
    #officialCloseRow = aggregateTable[(aggregateTable['TradeTime']==datetime.datetime(int(date[:4]),int(date[4:6]),int(date[6:8]),15,00,00)) & (aggregateTable['type']=='snapshot')]
    officialCloseRow = aggregateTable[aggregateTable['TradingPhaseCode_S']=='E0'].iloc[0:1,:]
    selfSnapshotDict = close_matching_engine.get_snapshot().snapshotDict    
    officialSnapshotDict = SnapshotDictSchema()
    officialSnapshotDict.initSeries(officialCloseRow.iloc[0])
    dif = officialSnapshotDict - selfSnapshotDict
    # 集合竞价的snapshot会出现价格为0的地方不要惊慌，这只是表示存在还没匹配的某一方向的量
    



if issueType in ['shkcbstock','shstock']:
    closeCallAuctionOrders = matching_engine.get_all_orders()
    # 重新初始化一个收盘集合竞价撮合引擎
    close_matching_engine = MatchingEngine(seed=456)
    closeCallAuctionAggregateTable = aggregateTable[(aggregateTable['TradeTime']>=closeCallAuctionTime) & ((aggregateTable['TradeTime']<=closeTime))]
    closeCallAuctionAggregateTable.sort_values(by=['TradeTime','BizIndex'],inplace=True)
    for ind,row in closeCallAuctionAggregateTable.iloc[:].iterrows():
        timestamp = row['TradeTime']
        match row['type']:
            case 'entrust':
                side = BUY_SELL_DICT[row['Side_E']]
                price = row['Price_E']
                size = row['OrderQty_E']
                timestamp = row['TradeTime']
                order_id = row['ApplSeqNum']
                thisOrder = LimitOrder(side=side, price=price, size=size, timestamp=timestamp, order_id=int(order_id),trader_id=int(order_id))
                closeCallAuctionOrders.add([thisOrder])  

            case 'trade':
                # 收盘集合竞价不允许撤单
                tradeQty = row['TradeQty_T']
                bidApplSeqNum = row['BidApplSeqNum']
                offerApplSeqNum = row['OfferApplSeqNum']
                closeCallAuctionOrders.trade(int(bidApplSeqNum), tradeQty)
                closeCallAuctionOrders.trade(int(offerApplSeqNum), tradeQty)

    closeCallAuctionTrades = close_matching_engine.match(timestamp=closeCallAuctionTime,orders=closeCallAuctionOrders)
    afterCallAuctionSnapshot = close_matching_engine.snapshot.summary()
    # 收盘snapshot的时间取决于交易所发布的时间，可能需要手工调整因此需要自动识别
    officialCloseRow = aggregateTable[(aggregateTable['TradeTime']==datetime.datetime(int(date[:4]),int(date[4:6]),int(date[6:8]),15,00,4)) & (aggregateTable['type']=='snapshot')]
    selfSnapshotDict = close_matching_engine.get_snapshot().snapshotDict    
    officialSnapshotDict = SnapshotDictSchema()
    officialSnapshotDict.initSeries(officialCloseRow.iloc[0])
    dif = officialSnapshotDict - selfSnapshotDict
    
