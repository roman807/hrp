from utils.data_loader import DataLoader

CONFIG = {
    "api": "alphavantage",
    "function": "TIME_SERIES_DAILY",
    "path": "sample_data/s_and_p500/",
    "symbols": "['IVV', 'TSLA', 'GOOG', 'FB', 'AMZN', 'KO', 'BX',]",
    "apikey": "JHC56UFM63VB6RUJ",
    "outputsize": "full",
    "datatype": "csv"
}
SAVE_DIR = 'data/sample_data/'


def main():
    data_loader = DataLoader(CONFIG, save_dir=SAVE_DIR)
    data_loader.load_data(save_as_csv=True, get_prices_and_returns=False)


if __name__ == '__main__':
    main()
