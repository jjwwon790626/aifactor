from collections import defaultdict
from datetime import datetime
from typing import cast

import pandas as pd
from pandera.typing import DataFrame
from pandera.typing import Series

from order_matching.order import Order
from order_matching.orders import Orders
from order_matching.schemas import OrderBookSummarySchema
from order_matching.schemas import SnapshotDictSchema
from order_matching.side import Side
import ast
OrderBookOrdersType = dict[float, Orders]


class OrderBook:
    """Order Book storage class."""

    def __init__(self) -> None:
        self.bids: OrderBookOrdersType = defaultdict(Orders)
        self.offers: OrderBookOrdersType = defaultdict(Orders)
        self.orders_by_expiration: dict[datetime, Orders] = defaultdict(Orders)

    def append(self, incoming_order: Order) -> None:
        """Add one order to the order book.

        Parameters
        ----------
        incoming_order
            New order
        """
        orders = self._get_same_side_orders(incoming_order=incoming_order)
        orders[incoming_order.price].add(orders=[incoming_order])
        self.orders_by_expiration[incoming_order.expiration].add(orders=[incoming_order])

    def remove(self, incoming_order: Order) -> None:
        """Remove one order from the order book.

        Parameters
        ----------
        incoming_order
            Order to be removed
        """
        orders = self._get_same_side_orders(incoming_order=incoming_order)
        orders[incoming_order.price].remove(orders=[incoming_order])
        self.orders_by_expiration[incoming_order.expiration].remove(orders=[incoming_order])
        if len(orders[incoming_order.price]) == 0:
            orders.pop(incoming_order.price)
        if len(self.orders_by_expiration[incoming_order.expiration]) == 0:
            self.orders_by_expiration.pop(incoming_order.expiration)

    def summary(self) -> DataFrame[OrderBookSummarySchema]:
        """Summary of the order book as a pandas DataFrame.

        Returns
        -------
        DataFrame[OrderBookSummarySchema]
            Summary of the order book as a pandas DataFrame
        """
        bids = pd.DataFrame(
            {
                OrderBookSummarySchema.side: Side.BUY.name,
                OrderBookSummarySchema.price: self._get_bid_prices(),
                OrderBookSummarySchema.size: self._get_bid_sizes(),
                OrderBookSummarySchema.count: self._get_bid_counts(),
            }
        )
        offers = pd.DataFrame(
            {
                OrderBookSummarySchema.side: Side.SELL.name,
                OrderBookSummarySchema.price: self._get_offer_prices(),
                OrderBookSummarySchema.size: self._get_offer_sizes(),
                OrderBookSummarySchema.count: self._get_offer_counts(),
            }
        )
        return pd.concat([bids, offers], ignore_index=True).assign(
            **{OrderBookSummarySchema.count: lambda df: df[OrderBookSummarySchema.count].astype(int)}
        )
    
    @property
    def snapshotDict(self)->SnapshotDictSchema:
        snapshotDictSchema = SnapshotDictSchema()
        snapshotDictSchema.initList(self._get_bid_prices(),self._get_bid_sizes(),
                                    self._get_bid_counts(),self._get_offer_prices(),
                                    self._get_offer_sizes(),self._get_offer_counts())
        return snapshotDictSchema

    @property
    def current_price(self) -> float:
        """Current market price."""
        return (self.max_bid + self.min_offer) / 2

    def get_opposite_side_orders(self, incoming_order: Order) -> OrderBookOrdersType:
        """Get orders on the opposite side.

        If incoming order is sell, then bids are returned.
        If incoming order is buy, then offers are returned.

        Parameters
        ----------
        incoming_order

        Returns
        -------
        OrderBookOrdersType
        """
        match incoming_order.side:  # noqa E501
            case Side.SELL:
                return self.bids
            case Side.BUY:
                return self.offers

    def get_subset(self, expiration: datetime) -> Orders:
        """Get orders with given expiration time.

        Parameters
        ----------
        expiration

        Returns
        -------
        Orders
        """
        return self.orders_by_expiration[expiration]

    def matching_order_exists(self, incoming_order: Order) -> bool:
        """Check that matching order exists.

        Parameters
        ----------
        incoming_order

        Returns
        -------
        bool
        """
        match incoming_order.side:
            case Side.SELL:
                return incoming_order.price <= self.max_bid and len(self.bids) > 0
            case Side.BUY:
                return incoming_order.price >= self.min_offer and len(self.offers) > 0

    def get_matching_sorted_opposite_side_prices(self, incoming_order: Order) -> list[float]:
        """Get sorted prices of the matching orders on the opposite side.

        Parameters
        ----------
        incoming_order

        Returns
        -------
        list[float]
        """
        prices = self._get_sorted_opposite_side_prices(incoming_order=incoming_order)
        match incoming_order.side:
            case Side.SELL:
                return list(filter(lambda price: price >= incoming_order.price, prices))
            case Side.BUY:
                return list(filter(lambda price: price <= incoming_order.price, prices))

    def get_imbalance(self, price_range: float = 0.1) -> float:
        r"""Calculate order book imbalance.

        Order book imbalance indicator is defined as
        $$
        \rho(L)=\frac{V_L^b-V_L^a}{V_L^b+V_L^a}\in[-1,1],
        $$
        where
        $$
        V_L^b=\sum_i Q_i^b\cdot\mathbf{1}\left[P_i^b\geq P^M-L\right],
        $$
        and
        $$
        V_L^a=\sum_i Q_i^a\cdot\mathbf{1}\left[P_i^a\leq P^M+L\right],
        $$
        are the total bid/ask volumes within $L>0$ from the market price.
        It is said that the market is balanced when $\rho\approx0$.
        The market is imbalanced and there exists demand (supply) pressure when $\rho>0$ ($\rho<0$).
        As it is clear from the formula,
        the imbalance can be computed at any distance $L$ from the market price.
        The higher the distance, the deeper looks this indicator into the order book.

        Parameters
        ----------
        price_range
            Left/right from the current market price

        Returns
        -------
        float
            Market imbalance indicator
        """
        summary = self.summary()
        if summary.empty:
            return 0
        elif summary[summary[OrderBookSummarySchema.side] == Side.SELL.name].empty:
            return 1
        elif summary[summary[OrderBookSummarySchema.side] == Side.BUY.name].empty:
            return -1
        else:
            return self._get_non_trivial_imbalance(price_range=price_range)

    def _get_non_trivial_imbalance(self, price_range: float) -> float:
        schema = OrderBookSummarySchema
        upper_bound = self.current_price + price_range
        lower_bound = self.current_price - price_range
        summary_subset = self.summary().pipe(lambda df: df[df[schema.price].between(lower_bound, upper_bound)])
        summary_subset = cast(pd.DataFrame, summary_subset)
        buy_volume = summary_subset.loc[summary_subset[schema.side] == Side.BUY.name, schema.size].sum()
        sell_volume = summary_subset.loc[summary_subset[schema.side] == Side.SELL.name, schema.size].sum()
        if buy_volume + sell_volume > 0:
            return (buy_volume - sell_volume) / (buy_volume + sell_volume)
        else:
            return 0

    def _get_same_side_orders(self, incoming_order: Order) -> OrderBookOrdersType:
        match incoming_order.side:
            case Side.SELL:
                return self.offers
            case Side.BUY:
                return self.bids

    def _get_sorted_opposite_side_prices(self, incoming_order: Order) -> list[float]:
        is_sell_side = incoming_order.side == Side.SELL
        return sorted(self.get_opposite_side_orders(incoming_order=incoming_order).keys(), reverse=is_sell_side)

    def _get_bid_prices(self) -> list[float]:
        return self._get_order_prices(orders=self.bids)

    def _get_offer_prices(self) -> list[float]:
        return self._get_order_prices(orders=self.offers)

    def _get_bid_sizes(self) -> list[float]:
        return self._get_order_sizes(orders=self.bids, prices=self._get_bid_prices())

    def _get_offer_sizes(self) -> list[float]:
        return self._get_order_sizes(orders=self.offers, prices=self._get_offer_prices())

    def _get_bid_counts(self) -> list[int]:
        return self._get_order_counts(orders=self.bids, prices=self._get_bid_prices())

    def _get_offer_counts(self) -> list[int]:
        return self._get_order_counts(orders=self.offers, prices=self._get_offer_prices())

    @staticmethod
    def _get_order_prices(orders: OrderBookOrdersType) -> list[float]:
        return sorted(orders.keys())

    @staticmethod
    def _get_order_sizes(orders: OrderBookOrdersType, prices: list[float]) -> list[float]:
        return [sum(order.size for order in orders[price]) for price in prices]

    @staticmethod
    def _get_order_counts(orders: OrderBookOrdersType, prices: list[float]) -> list[int]:
        return [len(orders[price]) for price in prices]

    @property
    def max_bid(self) -> float:
        """Maximum bid price."""
        if self.bids:
            return max(self.bids.keys())
        else:
            return 0.0

    @property
    def min_offer(self) -> float:
        """Minimum offer price."""
        if self.offers:
            return min(self.offers.keys())
        else:
            return float("inf")

    

  
    # @property
    # def summary_dict()    
#     def orderBookSummarySchema(self)-> OrderBookSummarySchema:
        
    

# snapshot = snapshot.iloc[1000]
#     def __init__(self, snapshot:Series):
