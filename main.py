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
    dist_mat = np.sqrt(.5 * (1 - cor_mat))
    link = sch.linkage(dist_mat, 'single')
    a=1
    # continue here


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
