from collections import OrderedDict, namedtuple
from itertools import product


class RunBuilder:
    @staticmethod
    def get_runs(params: OrderedDict) -> list:
        Run = namedtuple("Run", params.keys())
        runs = []

        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


params = OrderedDict(lr=[0.01, 0.001], batch_size=[1000, 10000])
runs = RunBuilder.get_runs(params)
print(runs)
print(params.values())
print(runs[0])
print(runs[0].lr, runs[0].batch_size)