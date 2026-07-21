import time
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# signal_validate: 
#   装饰器， 用来对模型输入进行检验。
def signal_validate(function):
    def _wrapper(*args, **kwargs):
        
        X = args[0]
        y = args[1]
        assert(len(X) == len(y))
        if kwargs.get("verbose"):
            start_time = time.time()
            result = function(*args, **kwargs)
            cost_time = time.time() - start_time
            print(f"{len(X)} data processed, {cost_time} secs used")
        return function(*args, **kwargs)
    return _wrapper

@signal_validate
def signal_linear(X, y, verbose=False, params={}):
    
    model = LinearRegression(**params)
    model = model.fit(X, y)
    pre_y = model.predict(X)

    return model, pre_y

@signal_validate
def signal_randomforest(X, y, verbose=False, params={}):

    model = RandomForestRegressor(**params)
    model = model.fit(X, y)
    pre_y = model.predict(X)

    return model, pre_y