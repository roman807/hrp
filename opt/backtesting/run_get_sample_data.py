import pandas as pd
import os

from utils.data_loader import DataLoader

PATH_UNIVERSE = 'sample_data/s_and_p500_tickers_v2.csv'
PATH_SAVE = 'sample_data/s_and_p500/'
# data_config['api']
DATA_CONFIG = {
    "api": "alphavantage",
    "function": "TIME_SERIES_DAILY",
    # "symbols": "['AAPL', 'GOOG', 'FB', 'IVV']",
    "apikey": "JHC56UFM63VB6RUJ",
    "outputsize": "full",
    "datatype": "csv"
}


def main():
    tickers = pd.read_csv(PATH_UNIVERSE, header=None)[0].values
    DATA_CONFIG['symbols'] = tickers#[:5]
    data_loader = DataLoader(
        data_config=DATA_CONFIG,
        universe=tickers,
        save_dir=PATH_SAVE)
    data_loader.load_data(get_returns=False, save_as_csv=True)


    # for f in os.listdir('sample_data/s_and_p500/'):
    #     pd.read_csv('sample_data/s_and_p500/' + f).iloc[:, :6].to_csv('sample_data/s_and_p500/' + f, index=False)



if __name__ == '__main__':
    main()
