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
        self.risk_appetite_scaled = None

    def get_expected_returns(self, conf):
        if not conf['consider_returns']:
            return None
        else:
            return pd.Series(conf['expected_returns'])[self.market_data.universe].reset_index(drop=True)

    def utility(self, w):
        risk_term = np.linalg.multi_dot([w, self.market_data.cov_mat, w]) if self.consider_returns else 0
        return_term = np.dot(self.exp_returns, w)
        return 1000 * risk_term - self.risk_appetite_scaled * return_term

    def scale_risk_apetite(self, w):
        risk_term = np.linalg.multi_dot([w, self.market_data.cov_mat, w])
        return_term = np.dot(self.exp_returns, w)
        self.risk_appetite_scaled = self.risk_appetite #* (risk_term / return_term)

    def optimize(self):
        init_w = np.array(len(self.market_data.universe) * [1/len(self.market_data.universe)])
        self.scale_risk_apetite(init_w)
        bounds = sco.Bounds(self.min_weight_constraints.values.tolist(), self.max_weight_constraints.values.tolist())
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        result = sco.minimize(self.utility, init_w, bounds=bounds, constraints=constraints, method='SLSQP')
        self.weights = pd.Series([np.round(i, 5) for i in result.x], index=self.market_data.universe)
        self.variance = np.round(np.linalg.multi_dot([np.array(result.x), self.market_data.cov_mat, np.array(result.x)]), 6)
        self.exp_return = np.round(sum(np.array(result.x) * np.array(self.exp_returns)), 6)
