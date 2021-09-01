import argparse


def get_data_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='anlys config', add_help=False)
    parser.add_argument(
        '-dc',
        '--data_conf',
        dest='data_conf',
        default='configs/dataconf_load_data_alphavantage_example.json',
        help='config for data'
    )
    return parser


def anlys_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='anlys config', add_help=False)
    parser.add_argument(
        '-dc',
        '--data_conf',
        dest='data_conf',
        default='configs/dataconf_anly_local_example.json',
        help='config for data'
    )
    return parser


def opt_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='portfolio optimization', add_help=False)
    parser.add_argument(
        '-c',
        '--conf',
        dest='conf',
        default='configs/optconf_hrp_example.json',
        help='config for strategy'
    )
    parser.add_argument(
        '-dc',
        '--data_conf',
        dest='data_conf',
        default='configs/dataconf_opt_local_example.json',
        help='config for data'
    )
    return parser
