from utils.utils import create_parser
from utils import data_loader
from functools import reduce
import pandas as pd
import datetime
import json
import importlib
import os
import ast


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
        ts = data[data['timestamp'] >= min_date][['timestamp', 'close']]
        ts = ts.set_index('timestamp')
        ts.columns = [symbol]
        timeseries.append(ts)

    reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), timeseries)
    # continue here


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
