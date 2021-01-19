import numpy as np
import pandas as pd
import datetime
import json
import time
from functools import reduce

from hierarchical_risk_parity import HierarchicalRiskParity
from mean_variance import MeanVariance
from utils.utils import create_parser, get_symbols
from utils.data_loader import DataLoader
from utils.constraints_builder import ConstraintsBuilder
from utils.market_data import MarketData


def get_config(config_json: str) -> dict:
    with open(config_json) as f:
        config = json.load(f)
    return config


def main():
    start_time = time.time()
    parser = create_parser()
    args = parser.parse_args()
    conf = get_config(args.conf)
    data_conf = get_config(args.data_conf)
    universe = get_symbols(data_conf['symbols'])
    constraints = ConstraintsBuilder(conf, universe)
    min_date = datetime.date.today() - datetime.timedelta(days=365*conf['yrs_look_back'])

    all_ts = []
    for symbol in universe:
        print(' ... load:', symbol)
        data_conf_ = data_conf.copy()
        data_conf_['symbol'] = symbol
        data_loader = DataLoader(data_conf_)
        all_ts.append(data_loader.get_returns(min_date))

    df_returns = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), all_ts)
    market_data = MarketData(df_returns)
    if conf['max_number_instruments'] < len(market_data.universe):
        hrp = HierarchicalRiskParity(market_data, constraints, conf)
        hrp.optimize()
        reduced_universe = hrp.weights.sort_values(ascending=False).index.tolist()[:conf['max_number_instruments']]
        market_data.reduce_market_data(reduced_universe)
        constraints.ajdust_universe(reduced_universe)

    # opt = HierarchicalRiskParity(market_data, constraints, conf)
    opt = MeanVariance(market_data, constraints, conf)
    opt.optimize()
    print('\noptimized in {} seconds'.format(np.round(time.time() - start_time, 5)))
    # constraints.check_constraints(opt.weights)
    print('optimal allocation:')
    print(opt.weights)
    print('variance:', opt.variance)
    print('expected return:', opt.exp_return)


if __name__ == '__main__':
    main()
