from .models import signal_SGDlinear, signal_linear, signal_randomforest
import pandas as pd
import numpy as np

def linear_model(kwargs):

    description = {

        "features": "DataFrame",
        "target": "Series or numpy.array",
        "verbose": bool,
        "params":{
            "please check sklearn.linear_model.LinearRegression"
        }
    }

    if kwargs is None:
        print(description)
        return

    features = kwargs.get("features")
    if features is None:
        raise AttributeError("{features} not in inputs")

    target = kwargs.get("target")
    if target is None:
        raise AttributeError("{target} not in inputs")
    if type(target) in [pd.DataFrame, pd.Series]:
        target = np.reshape(target.values, (-1, 1))

    verbose = kwargs.get("verbose", False)
    params = kwargs.get("params", {})

    return signal_linear(features, target, verbose=verbose, params=params)


def randomforest_model(kwargs):

    description = {

        "features": "DataFrame",
        "target": "Series or numpy.array",
        "verbose": bool,
        "params":{
            "please check sklearn.ensemble.RandomForestRegressor"
        }
    }

    if kwargs is None:
        print(description)
        return

    features = kwargs.get("features")
    if features is None:
        raise AttributeError("{features} not in inputs")

    target = kwargs.get("target")
    if target is None:
        raise AttributeError("{target} not in inputs")
    if type(target) in [pd.DataFrame, pd.Series]:
        target = np.reshape(target.values, (-1, 1))

    verbose = kwargs.get("verbose", False)
    params = kwargs.get("params", {})

    return signal_randomforest(features, target, verbose=verbose, params=params)


def SGDlinear_model(kwargs):

    description = {

        "features": "DataFrame",
        "target": "Series or numpy.array",
        "verbose": bool,
        "params":{
            "please check sklearn.ensemble.SGDlinear_model"
        }
    }

    if kwargs is None:
        print(description)
        return

    features = kwargs.get("features")
    if features is None:
        raise AttributeError("{features} not in inputs")

    target = kwargs.get("target")
    if target is None:
        raise AttributeError("{target} not in inputs")
    if type(target) in [pd.DataFrame, pd.Series]:
        target = np.reshape(target.values, (-1, 1))

    verbose = kwargs.get("verbose", False)
    params = kwargs.get("params", {})

    return signal_SGDlinear(features, target, verbose=verbose, params=params)
