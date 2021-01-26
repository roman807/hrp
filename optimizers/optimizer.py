import numpy as np
import pandas as pd
from abc import abstractmethod

from utils.market_data import MarketData
from utils.constraints_builder import ConstraintsBuilder


class Optimizer:

    def __init__(self, market_data: MarketData, constraints: ConstraintsBuilder, conf: dict):
        self.market_data = market_data
        self.consider_returns = conf['consider_returns']
        self.risk_appetite = conf['risk_appetite']
        self.exp_returns = self.get_expected_returns(conf)
        self.min_weight_constraints = constraints.min_weight_constraints.reset_index(drop=True)
        self.max_weight_constraints = constraints.max_weight_constraints.reset_index(drop=True)
        self.weights = None
        self.variance = None
        self.exp_return = None

    def get_expected_returns(self, conf):
        if not self.consider_returns:
            return None
        else:
            return pd.Series(conf['expected_returns'])[self.market_data.universe].reset_index(drop=True)

    @abstractmethod
    def optimize(self):
        pass

    def evaluate(self):
        self.variance = np.round(np.linalg.multi_dot([self.weights, self.market_data.cov_mat, self.weights]), 6)
        if self.consider_returns:
            self.exp_return = np.round(sum(np.array(self.weights) * np.array(self.exp_returns)), 6)

    def print_result(self):
        # self.variance = np.round(np.linalg.multi_dot([self.weights, self.market_data.cov_mat, self.weights]), 6)
        # self.exp_return = np.round(sum(np.array(self.weights) * np.array(self.exp_returns)), 6)
        print('optimal allocation:')
        print(self.weights)
        print('variance:', self.variance)
        print('expected return:', self.exp_return)

