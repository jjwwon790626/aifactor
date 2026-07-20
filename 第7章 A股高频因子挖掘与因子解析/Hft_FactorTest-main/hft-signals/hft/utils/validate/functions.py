import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator

from .numeric_stats import numeric_stats

from .plot_stats import feature_partition_return
from .plot_stats import others_corr
from .plot_stats import feature_hist
from .plot_stats import return_corr
from .plot_stats import partial_corr_hist
from statsmodels.graphics.tsaplots import plot_pacf

from .MDI import MDI
from .PFV import PFV

def plot_stats(kwargs=None):

    if not kwargs:
        print("""input: {
            "datas": {
                    "orderbook": DataFrame,
                },
            "feature": Series,
            "other_features": DataFrame(optional),
            "params": {
                    "partition_return_lags": int(default 500),
                    "figsize": [float, float](default [15, 7])
                }
            }
            """)
        return

    feature = kwargs.get("feature")
    orderbook = kwargs.get("datas", {}).get("orderbook")
    other_features = kwargs.get("other_features")

    fig_amount = 6

    if feature is None:
        raise AttributeError("(feature) is not found in inputs")
    if orderbook is None:
        raise AttributeError("(orderbook) is not found in inputs")
    if other_features is None:
        fig_amount = 5

    ## create axes and figs
    width, height = kwargs.get("params", {}).get("figsize", [None, None])
    if not width or not height:
        width, height = 15, 7
    fig, axes = plt.subplots(fig_amount, 1, figsize=[width, height*fig_amount])

    ## feature_partition_corr
    partition_return_lag = kwargs.get("params", {}).get("partition_return_lags", None)
    if not partition_return_lag:
        partition_return_lag = 500
    feature_partition_return(feature, orderbook, plot=True, price_lag=partition_return_lag, fig=axes[0])
    axes[0].title.set_text("Feature Partition Return")
    
    ## feature_hist
    feature_hist(feature, plot=True, fig=axes[1])
    axes[1].title.set_text("Feature Histogram")
    
    ## return_corr
    return_corr(orderbook=orderbook, feature=feature, plot=True, fig=axes[2])
    axes[2].title.set_text("Return Correlation for Lags")

    ## pacf_plot
    plot_pacf(feature, ax=axes[3], lags=15)

    ## partial_return_corr
    partial_corr_hist(feature, orderbook, plot=True, fig=axes[4])
    axes[4].title.set_text("Partial Return Correlation for Lags")

    ## others_corr
    if other_features:
        others_corr(feature, other_features, plot=True, fig=axes[5])
        axes[5].title.set_text("Correlation with Other Features")

    return fig, axes