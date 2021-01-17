import numpy as np
import pandas as pd
import datetime
import json
from functools import reduce

from hierarchical_risk_parity import HierarchicalRiskParity
from utils.utils import create_parser, get_symbols
from utils import data_loader


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


def main():
    parser = create_parser()
    args = parser.parse_args()
    conf = get_config(args.conf)
    data_conf = get_config(args.data_conf)
    symbols = get_symbols(data_conf['symbols'])
    min_date = datetime.date.today() - datetime.timedelta(days=365*conf['yrs_look_back'])

    all_ts = []
    for symbol in symbols:
        print(' ... load:', symbol)
        data_config_ = data_conf.copy()
        data_config_['symbol'] = symbol
        ts = get_time_series(data_config_, min_date)
        all_ts.append(ts)

    df_returns = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), all_ts)
    cor_mat, cov_mat = np.corrcoef(df_returns.T), np.cov(df_returns.T)
    hrp = HierarchicalRiskParity(cor_mat, cov_mat, symbols)
    hrp.optimize()
    print(hrp.weights)


if __name__ == '__main__':
    main()
