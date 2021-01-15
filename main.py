import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import datetime
import json
import os
import ast
from functools import reduce

from utils.utils import create_parser
from utils import data_loader


def get_config(config_json: str) -> dict:
    with open(config_json) as f:
        config = json.load(f)
    return config


def get_symbols(symbols_config: str) -> list:
    if os.path.isfile(symbols_config):
        with open(symbols_config, 'r') as f:
            symbols = f.read().split('\n')
    else:
        symbols = ast.literal_eval(symbols_config)
    return symbols


def get_quasi_diag(link: np.array) -> list:
    """
    Snippet 16.2, page 229
    :param link: linkage matrix
    :return:
    """
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)   # make space
        df0 = sort_ix[sort_ix >= num_items]   # find clusters
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]   # item 1
        df0 = pd.Series(link[j, 1], index=i+1)
        sort_ix = sort_ix.append(df0)   # item 2
        sort_ix.sort_index(inplace=True)   # re-sort
        sort_ix.index = range(sort_ix.shape[0])   # re-index
    return sort_ix.tolist()


def get_cluster_var(cov, c_items):
    cov_ = cov[c_items][:, c_items]
    w = np.diag(cov_) ** -1 / np.trace(np.diag(np.diag(cov_) ** -1))
    return np.linalg.multi_dot([w, cov_, w])


def get_rec_bipart(cov, sort_ix):
    """
    Snippet 16.3, page 230
    :param cov:
    :param sort_ix:
    :return:
    """
    # to implement CONTINUE HERE
    w = pd.Series(1, index=sort_ix)
    c_items = [sort_ix]
    while len(c_items) > 0:
        # c_items = [i[j:k] for i in c_items for j, k in ((0, len(i)/2), (len(i)/2, len(i))) if len(i) > 1]
        c_items = [i[j:k] for i in c_items for j, k in zip([0, int(len(i)/2)], [int(len(i)/2), len(i)]) if len(i) > 1]
        for i in range(0, len(c_items), 2):
            c_items_0 = c_items[i]   # cluster 1
            c_items_1 = c_items[i+1]   # cluster 2
            c_var_0 = get_cluster_var(cov, c_items_0)
            c_var_1 = get_cluster_var(cov, c_items_1)
            alpha = 1 - c_var_0 / (c_var_0 + c_var_1)
            w[c_items_0] *= alpha
            w[c_items_1] *= 1 - alpha
    return w


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    conf = get_config(args.conf)
    data_conf = get_config(args.data_conf)
    symbols = get_symbols(data_conf['symbols'])

    min_date = datetime.date.today() - datetime.timedelta(days=365*conf['yrs_look_back'])

    timeseries = []
    for symbol in symbols:
        # if symbol=='KO':
        #     a=1
        print(' ... analyze:', symbol)
        data_config_ = data_conf.copy()
        data_config_['symbol'] = symbol
        data_loader_ = data_loader.DataLoader(data_config_)
        data = data_loader_.get_data()
        data['returns'] = [0.0] + (np.log(np.array(data['close'])[1:] / np.array(data['close'])[:-1])).tolist()
        ts = data[data['timestamp'] >= min_date][['timestamp', 'returns']]
        ts = ts.set_index('timestamp')
        ts.columns = [symbol]
        timeseries.append(ts)

    df_returns = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), timeseries)

    cor_mat = np.corrcoef(df_returns.T)
    cov = np.cov(df_returns.T)
    dist_mat = np.sqrt(.5 * (1 - cor_mat))
    link = sch.linkage(dist_mat, 'single')
    sort_ix = get_quasi_diag(link)
    w = get_rec_bipart(cov, sort_ix)
    a=1
    # continue here -> implement recursive bisection


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
