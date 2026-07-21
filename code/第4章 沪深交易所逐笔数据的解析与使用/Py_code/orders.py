from __future__ import annotations

from dataclasses import asdict
from typing import Generator, Iterator, Sequence

import pandas as pd
from pandera.typing import DataFrame

from order_matching.order import Order
from order_matching.schemas import OrderDataSchema


class Orders:
    """Storage class for orders.

    Parameters
    ----------
    orders
    """

    def __init__(self, orders: Sequence[Order] | None = None) -> None:
        self.orders = list() if orders is None else list(orders)
        self._sort_orders_inplace()

    def add(self, orders: list[Order]) -> None:
        """Add new orders to the storage and sort them by timestamp.

        Parameters
        ----------
        orders
        """
        self.orders.extend(orders)
        self._sort_orders_inplace()

    def dequeue(self) -> Order:
        """Get the earliest order and remove it from the storage.

        Returns
        -------
        Order
            Earliest order
        """
        return self.orders.pop(0)

    def remove(self, orders: list[Order]) -> None:
        """Remove a list of orders from the storage matching by `order_id`.

        Partially executed orders may have their properties changed, hence matching is done only based on the id.

        Parameters
        ----------
        orders
        """

        for order_to_remove in orders:
            if order_to_remove.order_id in self._order_ids:
                self.orders.remove(self._get_order(order_id=order_to_remove.order_id))

    def to_frame(self) -> DataFrame[OrderDataSchema]:
        """Get pandas DataFrame with all orders in the storage.

        Returns
        -------
        DataFrame[OrderDataSchema]
        """
        if len(self.orders) == 0:
            return pd.DataFrame()
        else:
            return pd.DataFrame.from_records([asdict(order) for order in self.orders]).assign(
                **{
                    OrderDataSchema.side: lambda df: df[OrderDataSchema.side].astype(str),
                    OrderDataSchema.execution: lambda df: df[OrderDataSchema.execution].astype(str),
                    OrderDataSchema.status: lambda df: df[OrderDataSchema.status].astype(str),
                    OrderDataSchema.timestamp: lambda df: pd.to_datetime(
                        df[OrderDataSchema.timestamp], errors="coerce"
                    ),
                    OrderDataSchema.expiration: lambda df: pd.to_datetime(
                        df[OrderDataSchema.expiration], errors="coerce"
                    ),
                }
            )

    @property
    def is_empty(self) -> bool:
        """Check if the storage is empty."""
        return len(self.orders) == 0

    def __add__(self, other: Orders) -> Orders:
        orders = Orders()
        orders.add(orders=self.orders)
        orders.add(orders=other.orders)
        return orders

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Orders):
            return NotImplemented
        else:
            return self.orders == other.orders

    def __iter__(self) -> Iterator[Order]:
        return iter(self.orders)

    def __next__(self) -> Generator[Order, None, None]:
        yield next(self.__iter__())

    def __len__(self) -> int:
        return len(self.orders)

    def _sort_orders_inplace(self) -> None:
        self.orders.sort(key=lambda order: order.timestamp)

    @property
    def _order_ids(self) -> list[str]:
        return [order.order_id for order in self.orders]

    def _get_order(self, order_id: str) -> Order:
        orders = [order for order in self.orders if order.order_id == order_id]

        if len(orders)>1:
            print([x.order_id for x in orders])
        assert len(orders) <= 1
        return orders[0]

    def _deduce_size(self, order_id:str, deduce_size:float) -> None:
        order_id = str(order_id)
        orders = [order for order in self.orders if order.order_id == order_id]

        assert len(orders) <= 1
        thisOrder = orders[0]
        thisOrder.size -= deduce_size

        if thisOrder.size<=0:
            #print(thisOrder.order_id)
            self.orders.remove(thisOrder)



    def trade(self,seqNum:str,tradeQty:float)-> None:
        # 最终都会将大于集合竞价
        #print(seqNum)
        self._deduce_size(str(seqNum),tradeQty)
        

    def cancel_order(self, orderID: str):
        '''
        

        Parameters
        ----------
        orderID : str
            cancel a order ID is string

        Returns
        -------
        No return

        '''
        orderID = str(orderID)
        for order in list(self.orders):
            if order.order_id == orderID:
                self.remove([order])
            
