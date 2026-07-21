from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("order-matching")
except PackageNotFoundError:
    # package is not installed
    pass
