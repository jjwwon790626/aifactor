from enum import Enum


class CustomEnum(Enum):
    """Custom enumerator.

    Used across the library as a base class for enumeration objects.
    """

    def __lt__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return self.value < other.value
        else:
            return NotImplemented

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name
