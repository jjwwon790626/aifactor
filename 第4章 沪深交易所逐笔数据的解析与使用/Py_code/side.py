from order_matching.custom_enum import CustomEnum


class Side(CustomEnum):
    """Order side. Buy or sell."""

    BUY = 0
    SELL = 1
