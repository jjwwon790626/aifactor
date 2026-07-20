from order_matching.custom_enum import CustomEnum


class Status(CustomEnum):
    """Order status."""

    OPEN = 0
    CANCEL = 1
