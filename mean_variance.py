import numpy as np
import pandas as pd
import scipy.optimize as sco

from utils.market_data import MarketData
from utils.constraints_builder import ConstraintsBuilder


class MeanVariance:

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
        if not conf['consider_returns']:
            return None
        else:
            return pd.Series(conf['expected_returns'])[self.market_data.universe].reset_index(drop=True)

    def utility(self, w):
        # return np.linalg.multi_dot([w, cov, w])
        return 1000 * np.linalg.multi_dot([w, self.market_data.cov_mat, w])

    def optimize(self):
        args = self.market_data.cov_mat
        initial_weights = np.array(len(self.market_data.universe) * [1/len(self.market_data.universe)])
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        result = sco.minimize(self.utility, initial_weights, constraints=constraints, method='SLSQP')
        self.weights = pd.Series(result.x, index=self.market_data.universe)
        self.variance = np.round(np.linalg.multi_dot([np.array(result.x), self.market_data.cov_mat, np.array(result.x)]), 6)
        self.exp_return = np.round(sum(np.array(result.x) * np.array(self.exp_returns)), 6)
        # next: add constraints, consider returns