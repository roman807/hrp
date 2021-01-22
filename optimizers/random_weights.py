import pandas as pd
import numpy as np

from optimizers.optimizer import Optimizer
from utils.market_data import MarketData
from utils.constraints_builder import ConstraintsBuilder


class RandomWeights(Optimizer):

    def __init__(self, market_data: MarketData, constraints: ConstraintsBuilder, conf: dict):
        super().__init__(market_data, constraints, conf)

    def optimize(self):
        # todo: add constraints
        w = np.random.uniform(0, 1, len(self.market_data.universe))
        self.weights = pd.Series(w / sum(w), index=self.market_data.universe)
