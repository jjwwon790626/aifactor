import json


def get_type(_type):
    mapper = {
        "FLOAT": float,
        "INTEGER": int,
        "STRING": str,
        "OBJECT": object,
        "ARRAY": list
    }
    if _type in mapper:
        return mapper[_type]
    raise AttributeError(f"type ({_type}) not found")


def get_validate_info(param_v, validate={}, name=''):#validate 校验值与约束的关系
  for k, v in validate.items(): # 对各种校验条件的检验
    if k == 'not_null' and v == True:
        if not param_v:
            return f"required calculate params ({name}) not found in inputs"
    if k == 'range':
        min_v = v.get('min')
        if min_v is not None and min_v > param_v:
            return f"{name} ({param_v}) < min({min_v})"
        max_v = v.get('max')
        if max_v is not None and max_v < param_v:
            return f"{name} ({param_v}) > max({max_v})"
    if k == 'type':
        if str(type(param_v)) != get_type(v) and type(param_v) != get_type(v):
            return f"{name} ({param_v}) should be {str(v)} type"
                            

class IFactor:
    describe = {
        "name": "interface_factor",
        "datas": [
            "depth5",
            "trade",
            "bookticker"
        ],
        "params": [{
            "name": "window_length",
            "default_value": 1,
            "validate": {
                "not_null": True,
                "range": {
                    "min": 0,
                    "max": 10
                }
            }
        }],
        "description": """
        this is a interface of factor.
        """
    }
    def __init__(self, **kwargs):
        
        if not kwargs:
            self.__class__.__print_describe(self)
            
        self.__describe = self.__class__.describe
        self.__check_params(kwargs)
        
    def __new__(cls, *args, **kwargs):
        
        if not kwargs:
            cls.__print_describe(cls)
            return 
        
        cls.__check_params(cls, **kwargs)
        return cls.main(cls, **kwargs)
        
    def __check_params(cls, **kwargs):
        
        datas = kwargs.get("datas")
        params = kwargs.get("params")
        
        if datas is None:
            raise AttributeError("params (datas) not found in inputs")
        if params is None:
            raise AttributeError("params (params) not found in inputs")
        params_configs = cls.describe["params"]
        for params_config in params_configs:
            validate = params_config.get("validate")
            name = params_config.get('name')
            if validate:
                param_v = params.get(name)
                validate_info = get_validate_info(param_v, validate, name)
                if validate_info: 
                    raise validate_info
                            
        required_datas = cls.describe["datas"]
        for data in required_datas:
            if datas.get(data) is None:
                raise AttributeError(f"required data ({data}) not found in inputs")

    def __print_describe(cls):
        print(json.dumps(cls.describe, indent=4))
    
    def main(cls, **kwargs):
        raise NotImplementedError("")
