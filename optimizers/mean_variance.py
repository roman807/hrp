import numpy as np
import pandas as pd
import scipy.optimize as sco

from utils.market_data import MarketData
from utils.constraints_builder import ConstraintsBuilder
from optimizers.optimizer import Optimizer


class MeanVariance(Optimizer):

    def __init__(self, market_data: MarketData, constraints: ConstraintsBuilder, conf: dict):
        super().__init__(market_data, constraints, conf)

    def utility_no_ret(self, w):
        return 1000 * np.linalg.multi_dot([w, self.market_data.cov_mat, w])

    def utility(self, w):
        risk_term = np.linalg.multi_dot([w, self.market_data.cov_mat, w])
        return_term = np.dot(self.exp_returns, w) #if self.consider_returns else 0
        return 1000 * risk_term - self.risk_appetite_scaled * return_term

    def scale_risk_apetite(self, w):
        risk_term = np.linalg.multi_dot([w, self.market_data.cov_mat, w])
        return_term = np.dot(self.exp_returns, w)
        self.risk_appetite_scaled = self.risk_appetite #* (risk_term / return_term)

    def optimize(self):
        init_w = np.array(len(self.market_data.universe) * [1/len(self.market_data.universe)])
        bounds = sco.Bounds(self.min_weight_constraints.values.tolist(), self.max_weight_constraints.values.tolist())
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        if self.consider_returns:
            self.scale_risk_apetite(init_w)
            result = sco.minimize(self.utility, init_w, bounds=bounds, constraints=constraints, method='SLSQP')
        else:
            result = sco.minimize(self.utility_no_ret, init_w, bounds=bounds, constraints=constraints, method='SLSQP')
        self.weights = pd.Series([np.round(i, 5) for i in result.x], index=self.market_data.universe)
