import argparse
import os
import ast


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='ts forecasting', add_help=False)
    parser.add_argument(
        '-c',
        '--conf',
        dest='conf',
        default='configs/conf_default.json',
        help='config for strategy'
    )
    parser.add_argument(
        '-dc',
        '--data_conf',
        dest='data_conf',
        # default='../configs/config_sample_data.json',
        default='configs/conf_data_alphavantage.json',
        help='config for data'
    )
    return parser


def get_symbols(symbols_config: str) -> list:
    if os.path.isfile(symbols_config):
        with open(symbols_config, 'r') as f:
            symbols = f.read().split('\n')
    else:
        symbols = ast.literal_eval(symbols_config)
    return symbols

# def create_dir(path_: str) -> str:
#     if not os.path.exists(path_):
#         os.makedirs(path_)
#     return path_
