import numpy as np
import pandas as pd
import datetime
import json
import time
from functools import reduce

from hierarchical_risk_parity import HierarchicalRiskParity
from utils.utils import create_parser, get_symbols
from utils import data_loader, constraints_builder


def get_config(config_json: str) -> dict:
    with open(config_json) as f:
        config = json.load(f)
    return config


def get_time_series(data_config: dict, min_date: datetime.date) -> pd.Series:
    data_loader_ = data_loader.DataLoader(data_config)
    data = data_loader_.get_data()
    data['returns'] = [0.0] + (np.log(np.array(data['close'])[1:] / np.array(data['close'])[:-1])).tolist()
    ts = data[data['timestamp'] >= min_date][['timestamp', 'returns']]
    ts = ts.set_index('timestamp')
    ts.columns = [data_config['symbol']]
    return ts


def get_input_market_data(df_returns: pd.DataFrame, constraints: constraints_builder.ConstraintsBuilder, conf: dict):
    symbols = df_returns.columns.tolist()
    cor_mat, cov_mat = np.corrcoef(df_returns.T), np.cov(df_returns.T)
    if conf['max_number_instruments'] < len(symbols):
        hrp = HierarchicalRiskParity(cor_mat, cov_mat, symbols, constraints, conf)
        hrp.optimize()
        symbols_red = hrp.weights.sort_values(ascending=False).index.tolist()[:conf['max_number_instruments']]
        new_ind = [i for i in range(len(symbols)) if symbols[i] in symbols_red]
        symbols = symbols_red
        cov_mat = cov_mat.take(new_ind, axis=0).take(new_ind, axis=1)
        cor_mat = cor_mat.take(new_ind, axis=0).take(new_ind, axis=1)
    return cor_mat, cov_mat, symbols


def main():
    start_time = time.time()
    parser = create_parser()
    args = parser.parse_args()
    conf = get_config(args.conf)
    data_conf = get_config(args.data_conf)
    all_symbols = get_symbols(data_conf['symbols'])
    constraints = constraints_builder.ConstraintsBuilder(conf, all_symbols)
    min_date = datetime.date.today() - datetime.timedelta(days=365*conf['yrs_look_back'])

    all_ts = []
    for symbol in all_symbols:
        print(' ... load:', symbol)
        data_config_ = data_conf.copy()
        data_config_['symbol'] = symbol
        ts = get_time_series(data_config_, min_date)
        all_ts.append(ts)

    df_returns = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), all_ts)
    cor_mat, cov_mat, symbols = get_input_market_data(df_returns, constraints, conf)
    constraints.ajdust_symbols(symbols)

    hrp = HierarchicalRiskParity(cor_mat, cov_mat, symbols, constraints, conf)
    hrp.optimize()
    print('\noptimized in {} seconds'.format(np.round(time.time() - start_time, 5)))
    constraints.check_constraints(hrp.weights)
    print('optimal allocation:')
    print(hrp.weights)
    print('variance:', hrp.variance)
    print('expected return:', hrp.exp_return)


if __name__ == '__main__':
    main()
