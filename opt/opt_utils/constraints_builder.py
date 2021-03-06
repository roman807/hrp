import pandas as pd
import numpy as np


class ConstraintsBuilder:

    def __init__(self, data_config: dict, universe: list):
        self.data_config = data_config
        self.universe = universe
        self.max_n_instr = data_config['max_number_instruments']
        self.min_weight_constraints = self.get_min_weight_constraints()
        self.max_weight_constraints = self.get_max_weight_constraints()

    def get_min_weight_constraints(self):
        min_weights = pd.Series(0.0, index=self.universe)
        for s, w in self.data_config['min_weight_constraints'].items():
            min_weights[s] = w
        return min_weights

    def get_max_weight_constraints(self):
        max_weights = pd.Series(1.0, index=self.universe)
        for s, w in self.data_config['max_weight_constraints'].items():
            max_weights[s] = w
        return max_weights

    def ajdust_universe(self, new_universe):
        self.universe = new_universe
        self.min_weight_constraints = self.min_weight_constraints[self.universe]
        self.max_weight_constraints = self.max_weight_constraints[self.universe]

    def check_constraints(self, final_weights):
        assert np.round((final_weights - self.min_weight_constraints).dropna().min(), 2) >= 0, 'min weight constraint not satisfied'
        assert np.round((self.max_weight_constraints - final_weights).dropna().min(), 2) >= 0, 'max weight constraint not satisfied'
        assert len(final_weights) <= self.max_n_instr
        print('all constraints satisfied')
