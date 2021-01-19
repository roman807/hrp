import numpy as np
import pandas as pd


class MarketData:

    def __init__(self, df_returns: pd.DataFrame):
        self.universe = df_returns.columns.tolist()
        self.df_returns = df_returns
        self.cor_mat = np.corrcoef(df_returns.T)
        self.cov_mat = np.cov(df_returns.T)

    def reduce_market_data(self, reduced_universe: list):
        new_ind = [i for i in range(len(self.universe)) if self.universe[i] in reduced_universe]
        self.universe = reduced_universe
        self.cov_mat = self.cov_mat.take(new_ind, axis=0).take(new_ind, axis=1)
        self.cor_mat = self.cor_mat.take(new_ind, axis=0).take(new_ind, axis=1)
