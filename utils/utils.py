import argparse
import os
import ast
import json


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
