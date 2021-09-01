from utils.data_loader import DataLoader
from utils.parsers import get_data_parser
from utils.utils import get_config
import os


def main():
    parser = get_data_parser()
    args = parser.parse_args()
    data_conf = get_config(args.data_conf)

    if not os.path.exists(data_conf["target_dir"]):
        os.makedirs(data_conf["target_dir"])

    data_loader = DataLoader(data_conf)
    data_loader.load_data(save_as_csv=True, get_prices_and_returns=False)


if __name__ == '__main__':
    main()
