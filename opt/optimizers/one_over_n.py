import pandas as pd

from opt.optimizers.optimizer import Optimizer
from utils.market_data import MarketData
from opt.opt_utils.constraints_builder import ConstraintsBuilder


class OneOverN(Optimizer):

    def __init__(self, market_data: MarketData, constraints: ConstraintsBuilder, conf: dict):
        super().__init__(market_data, constraints, conf)

    def optimize(self):
        # todo: add constraints
        self.weights = pd.Series([1 / len(self.market_data.universe) for _ in self.market_data.universe],
                                 index=self.market_data.universe)
