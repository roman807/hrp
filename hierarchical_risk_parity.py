import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter
simplefilter("ignore", ClusterWarning)


class HierarchicalRiskParity:

    def __init__(self, cor_mat, cov_mat, symbols, constraints):
        self.cor_mat = cor_mat
        self.cov_mat = cov_mat
        self.symbols = symbols
        self.min_weight_constraints = constraints.min_weight_constraints.reset_index(drop=True)
        self.max_weight_constraints = constraints.max_weight_constraints.reset_index(drop=True)
        self.weights = None

    def get_cluster_var(self, cov, c_items) -> float:
        """
        calculate cluster variance defined as "w.T*cov*w" with weights assigned inversely proportional to variance
        :param cov: covariance matrix (with all constituents)
        :param c_items: constituents of cov matrix to consider
        :return: cluster variance
        """
        cov_ = cov[c_items][:, c_items]
        w = np.diag(cov_) ** -1 / np.trace(np.diag(np.diag(cov_) ** -1))
        return np.linalg.multi_dot([w, cov_, w])

    def get_quasi_diag(self, link: np.array) -> list:
        """
        Reorder rows & cols of cov matrix so that largest values lie along diagonal
        Snippet 16.2, page 229
        :param link: linkage matrix
        :return: sorted list of original items
        """
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)  # make space
            df0 = sort_ix[sort_ix >= num_items]  # find clusters
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]  # item 1
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = sort_ix.append(df0)  # item 2
            sort_ix.sort_index(inplace=True)  # re-sort
            sort_ix.index = range(sort_ix.shape[0])  # re-index
        return sort_ix.tolist()

    def get_rec_bipart(self, cov: np.array, sort_ix: list) -> pd.Series:
        """
        Assign weights to constituents according to inverse-variance allocation
        Snippet 16.3, page 230
        :param cov: covariance matrix
        :param sort_ix: sorted list of original items
        :return: optimal weight allocation
        """
        w = pd.Series(1, index=sort_ix)
        c_items = [sort_ix]
        while len(c_items) > 0:
            c_items = [i[j:k] for i in c_items for j, k in zip([0, int(len(i) / 2)], [int(len(i) / 2), len(i)]) if
                       len(i) > 1]
            for i in range(0, len(c_items), 2):
                c_items_0 = c_items[i]  # cluster 1
                c_items_1 = c_items[i + 1]  # cluster 2
                c_var_0 = self.get_cluster_var(cov, c_items_0)
                c_var_1 = self.get_cluster_var(cov, c_items_1)
                alpha = 1 - c_var_0 / (c_var_0 + c_var_1)

                # adjust for min constraints:
                min_w_0 = self.min_weight_constraints[c_items_0].sum()
                min_w_1 = self.min_weight_constraints[c_items_1].sum()
                if w[c_items_0].values[0] * alpha < min_w_0:
                    alpha = min_w_0 / w[c_items_0].values[0]
                if w[c_items_1].values[0] * (1-alpha) < min_w_1:
                    alpha = 1 - min_w_1 / w[c_items_1].values[0]

                # adjust for max constraints:
                max_w_0 = self.max_weight_constraints[c_items_0].sum()
                max_w_1 = self.max_weight_constraints[c_items_1].sum()
                if w[c_items_0].values[0] * alpha > max_w_0:
                    alpha = max_w_0 / w[c_items_0].values[0]
                if w[c_items_1].values[0] * (1-alpha) > max_w_1:
                    alpha = 1 - max_w_1 / w[c_items_1].values[0]

                w[c_items_0] *= alpha
                w[c_items_1] *= 1 - alpha
        return w

    def optimize(self):
        dist_mat = np.sqrt(.5 * (1 - self.cor_mat))
        link = sch.linkage(dist_mat, 'single')   # hierarchical/agglomerative clustering
        sort_ix = self.get_quasi_diag(link)
        w = self.get_rec_bipart(self.cov_mat, sort_ix)
        w.index = [self.symbols[i] for i in w.index]
        self.weights = w
