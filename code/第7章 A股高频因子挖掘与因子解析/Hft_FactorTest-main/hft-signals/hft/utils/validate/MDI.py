from typing import Any
from sklearn.ensemble import RandomForestRegressor
from hft.utils.target import filled_return
from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 随机森林回归模型分别训练Ask侧和Bid侧的收益预测，然后通过MDI方法量化每个特征的重要性

_default_randomforest_params = {
    "n_estimators": 100, # 树的数量
    "criterion": "mse",
    "max_features": int(1),
    "max_depth": 5,     # 限制深度防止过拟合
    "min_samples_split": 100, # 较大的分裂样本数
    "min_samples_leaf": 10,    # 叶节点最小样本数
    "n_jobs": -1,
    "verbose": 1
}

def MDI(features, ob, ask_model=None, bid_model=None):
    
    # preprocess
    features_amount = len(features)
    feature_names = features.columns
    
    if ask_model is None or bid_model is None:
        ask_model = RandomForestRegressor(**_default_randomforest_params)
        bid_model = RandomForestRegressor(**_default_randomforest_params)
        ask_y, bid_y = filled_return(ob)
        
        ask_model = ask_model.fit(features, ask_y)
        bid_model = bid_model.fit(features, bid_y)
        
    # impotances and these stds
    ask_impotances, bid_impotances = ask_model.feature_importances_, bid_model.feature_importances_
    ask_impo_std = np.std([tree.feature_importances_ for tree in ask_model.estimators_], axis=0)
    bid_impo_std = np.std([tree.feature_importances_ for tree in bid_model.estimators_], axis=0)
        
    ask_impotances = pd.Series(ask_impotances, index=feature_names)
    bid_impotances = pd.Series(bid_impotances, index=feature_names)
        
    plot_x = np.arange(len(feature_names))
    width = 0.4

    # barh plot 
    fig, ax = plt.subplots(figsize=[15,len(feature_names)])
    ask_bar = ax.barh(plot_x - width/2, ask_impotances.values, width, label="Ask", color="red")
    bid_bar = ax.barh(plot_x + width/2, bid_impotances.values, width, label="Bid", color="green")
    
    # add float to the bars
    ax.bar_label(ask_bar, padding=3)
    ax.bar_label(bid_bar, padding=3)

    # title and label
    ax.set_ylabel('Importances (MDI) ')
    ax.set_title('Ask and Bid RandomForestRegressor MDI')
    ax.set_yticks(plot_x)
    ax.set_yticklabels(feature_names)
    ax.legend()
    
    fig.tight_layout()
    
    plt.show()

    return ask_model, bid_model