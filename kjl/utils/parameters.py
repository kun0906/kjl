


class PARAM():

    def __init__(self, **kwargs):
        # ns = Namespace(**kwargs)      # or from argparse import Namespace
        for name in kwargs:
            setattr(self, name, kwargs[name])

    def add_param(self, **kwargs):
        for name in kwargs:
            setattr(self, name, kwargs[name])

    def __repr__(self):
        params_str = '-'.join([f'{k}:{v}' for k,v in sorted(self.__dict__.items(), key=lambda x:x[0])])
        # pprint(self.__dict__.items())
        return params_str


