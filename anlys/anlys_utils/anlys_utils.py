import argparse


def anlys_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='anlys config', add_help=False)
    # parser.add_argument(
    #     '-c',
    #     '--conf',
    #     dest='conf',
    #     default='opt/configs/conf_default.json',
    #     help='config for strategy'
    # )
    parser.add_argument(
        '-dc',
        '--data_conf',
        dest='data_conf',
        default='anlys/configs/conf_data_default.json',
        # default='opt/configs/conf_data_alphavantage.json',
        help='config for data'
    )
    return parser