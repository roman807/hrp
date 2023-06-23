import requests
import numpy as np
import pandas as pd
import os
from io import StringIO
from datetime import datetime
from functools import reduce
from collections import defaultdict
import time
from utils.utils import get_symbols

METHODS = {
    'alphavantage': 'load_data_alphavantage',
    'iex': 'load_data_iex',
    'local_data': 'load_local_data'
}
ALPHAVANTAGE_SLEEP_TIME = 60.1


class DataLoader:
    def __init__(self, data_config, from_date=None, to_date=None):
        self.data_config = data_config
        self.from_date = from_date
        self.to_date = to_date
        self.all_data = {}
        self.df_returns = defaultdict(pd.DataFrame)
        self.df_prices = defaultdict(pd.DataFrame)
        self.universe = get_symbols(data_config['symbols'])

    def load_data(self, save_as_csv=False, print_progress=True, adjust_for_split=True):
        """
        load datasets into all_data dict (key: ticker, value: dataset)
        if save_as_csv==True: CSVs will be stored locally for further analysis (e.g. run_anlys or run_opt with
        data previously loaded from run_get_data)
        """
        for symbol in self.universe:
            filename = symbol + '_' + str(datetime.today().date()) + '.csv'
            if save_as_csv:
                if 'target_dir' in self.data_config and filename in os.listdir(self.data_config['target_dir']):
                    print('skipping {}, {} already saved in {}'.format(symbol, filename, self.data_config['target_dir']))
                    continue
            if print_progress:
                print(' ... load:', symbol)
            data = getattr(DataLoader, METHODS[self.data_config['api']])(self, symbol)
            if 'split_coefficient' in data.columns and adjust_for_split:
                data = self.adjust_for_stock_split(data)
            if type(data) == type(None):
                continue
            if data.empty:
                continue
            else:
                self.all_data[symbol] = data
            if save_as_csv:
                self.all_data[symbol].to_csv(self.data_config['target_dir'] + filename, index=False)
                print('saved', symbol, 'in', self.data_config['target_dir'])

    def adjust_for_stock_split(self, data):
        # Todo: implement adjustment for stock-split
        return data

    def calculate_prices_and_returns(self):
        """
        calculate the following datasets (needed in run_anlys and run_opt)
        - df_returns: index: daily date, columns: one col per symbol with closing price
        - df_prices: index: daily date, columns: one col per symbol with daily return
        """
        all_returns, all_prices = [], []
        if not self.all_data:
            raise Exception("all_data is empty - load data first with the 'load_data'-function")

        for symbol in self.universe:
            if symbol not in self.all_data:
                print(f"skipping {symbol}, data not loaded")
                continue
            all_returns.append(self.get_returns(self.all_data[symbol], symbol))
            all_prices.append(self.get_prices(self.all_data[symbol], symbol))

        keys_to_remove = []
        for k, v in self.all_data.items():
            if not v['timestamp'].min() <= self.from_date:
                print('WARNING: remove {} from analysis -- history too short'.format(k))
                keys_to_remove.append(k)
        for k in keys_to_remove:
            self.all_data.pop(k)
        all_returns = [ts for ts in all_returns if ts.columns[0] not in keys_to_remove]
        all_prices = [ts for ts in all_prices if ts.columns[0] not in keys_to_remove]
        self.universe = [i for i in self.universe if i not in keys_to_remove]
        self.df_returns = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), all_returns)
        self.df_prices = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), all_prices)

    def get_datasets_for_mean_reversion(self, moving_avg_number_of_days, n_days_fwd=21, min_year=2010):
        """
        get dict with one dataset (value) per symbol (key). raw datasets must have been previously loaded with
        the 'load_data'-function. Returned datasets show current price vs moving averages and mean price in the
        coming 21 days
        """
        if not self.all_data:
            raise Exception("all_data is empty - load data first with the 'load_data'-function")

        datasets_mr = {}
        for symbol in self.all_data.keys():
            df_prices = self.all_data[symbol][['timestamp', 'close']].set_index('timestamp')
            df_prices = df_prices[df_prices.index >= datetime.strptime(str(min_year), '%Y').date()]
            moving_averages = []
            for n_days in moving_avg_number_of_days:
                df_moving_avg = df_prices.rolling(n_days).mean()
                df_moving_avg.columns = [f'moving_avg_{n_days}']
                moving_averages.append(df_moving_avg)
            df = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), moving_averages)
            df['fwd'] = np.nan
            df_fwd = df_prices.rolling(n_days_fwd).mean()
            df['fwd'][:-n_days_fwd] = df_fwd[n_days_fwd:]['close'].tolist()
            df = df.iloc[max(moving_avg_number_of_days)-1:-n_days_fwd, :]
            df = pd.merge(df, df_prices, left_index=True, right_index=True)
            for col in df.columns:
                df[col] = (df[col] / df['close']) - 1
            df.drop('close', axis=1, inplace=True)
            datasets_mr[symbol] = df
        return datasets_mr

    def get_datasets_for_scenario_prediction(
            self,
            margin=None,
            n_future_days_target=None,
            use_adjusted_close=True,
            with_target=True
    ):
        n_days_windows = [5, 10, 20, 63, 126, 251]
        # n_future_days_target = 63

        def map_future_return_to_target(x, margin):
            if x > margin:
                return 'up'
            if x < -margin:
                return 'down'
            return 'flat'

        if not self.all_data:
            raise Exception("all_data is empty - load data first with the 'load_data'-function")

        if use_adjusted_close:
            price_col = "adjusted_close"
        else:
            price_col = 'close'

        datasets_sp = {}
        for symbol in self.all_data.keys():
            df = self.all_data[symbol][['timestamp', price_col]]
            df_returns = pd.Series(self.get_returns(df, price_col=price_col, as_DataFrame=False))

            data_dict = {}
            data_dict['timestamp'] = df['timestamp'].apply(lambda x: pd.to_datetime(x))
            data_dict['price'] = df[price_col]
            data_dict['price_normalized'] = df[price_col] / df[price_col][0]
            for n_days in n_days_windows:
                # add moving averages:
                data_dict[f'moving_avg_{n_days}'] = df[price_col].rolling(n_days).mean()
                # add moving_average-based features:
                data_dict[f'current_over_moving_avg_{n_days}'] = (
                    df[price_col] / df[price_col].rolling(n_days).mean()
                )
                # add return-based features
                data_dict[f'return_over_{n_days}'] = (
                    np.concatenate((
                        np.array([np.nan] * n_days),
                        (df[price_col][n_days:].values - df[price_col][:-n_days].values) / df[price_col][:-n_days].values
                    ))
                )
                # add volatility-based features
                data_dict[f'vola_over_{n_days}'] = df_returns.rolling(n_days).std() * np.sqrt(n_days)

            # add target variable
            if with_target:
                data_dict[f'future_return'] = (
                    np.concatenate((
                        (
                            (df[price_col][n_future_days_target:].values - df[price_col][:-n_future_days_target].values)
                            / df[price_col][:-n_future_days_target].values
                        ),
                        np.array([np.nan] * n_future_days_target)
                    ))
                )
                df_symbol = pd.DataFrame(data_dict).iloc[max(n_days_windows):-n_future_days_target, :]
                df_symbol['target'] = df_symbol[f'future_return'].apply(lambda x: map_future_return_to_target(x, margin))
            else:
                df_symbol = pd.DataFrame(data_dict).iloc[max(n_days_windows):, :]
            datasets_sp[symbol] = df_symbol.set_index('timestamp', drop=True)
        return datasets_sp

    def get_returns(self, data, symbol=None, price_col='close', as_DataFrame=True):
        """
        :param data: dataframe with columns 'timestamp' and 'close'
        :param symbol: symbol of stock for return calculation. Only needed if 'as_DataFrame=True'
        :param price_col: column to perform return calculation on (either 'close' or 'adjusted_close')
        :param as_DataFrame: set to true if returns should be returned in a dataframe with timestamp as index
                             and 'symbol' as column name. If false, list of returns is returned instead
        :return: returns as list or DataFrame
        """
        returns = [0.0] + (np.log(np.array(data[price_col])[1:] / np.array(data[price_col])[:-1])).tolist()
        if not as_DataFrame:
            return returns
        else:
            data['returns'] = returns
            ts = data[(data['timestamp'] >= self.from_date) & (data['timestamp'] <= self.to_date)][['timestamp', 'returns']]
            ts = ts.set_index('timestamp')
            ts.columns = [symbol]
            return ts

    def get_prices(self, data, symbol):
        ts = data[(data['timestamp'] >= self.from_date) & (data['timestamp'] <= self.to_date)][['timestamp', 'close']]
        ts = ts.set_index('timestamp')
        ts.columns = [symbol]
        return ts

    def load_data_alphavantage(self, symbol) -> pd.DataFrame:
        """
        :return: daily prices from alphavantage: https://www.alphavantage.co/
        """
        def create_url(function, symbol, apikey, outputsize, datatype):
            return 'https://www.alphavantage.co/query?function=' + function + '&symbol=' + symbol +'&apikey=' + apikey + \
                   '&outputsize=' + outputsize + '&datatype=' + datatype

        url = create_url(
            function=self.data_config['function'],
            # symbol=self.data_config['symbol'],
            symbol=symbol,
            apikey=self.data_config['apikey'],
            outputsize=self.data_config['outputsize'],
            datatype=self.data_config['datatype'])
        response = requests.get(url)
        if "Error Message" in response.text:
            print('skipping', symbol, '(not available)')
            return pd.DataFrame()
        if '5 calls per minute' in response.text:
            print('reached alphavantage free API limit -> sleep for {} seconds'.format(round(ALPHAVANTAGE_SLEEP_TIME)))
            time.sleep(ALPHAVANTAGE_SLEEP_TIME)
            response = requests.get(url)
        data = pd.read_csv(StringIO(response.text))
        data['timestamp'] = data['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').date())
        data.sort_values(by='timestamp', ascending=True, inplace=True)
        return data

    def load_data_iex(self, symbol) -> pd.DataFrame:
        """
        :return: daily prices from https://iexcloud.io/
        """
        def create_url(symbol, range_, datatype, token):
            return 'https://cloud.iexapis.com/stable/stock/' + symbol + '/chart/' + range_ + '?format=' + datatype + \
                   '&token=' + token
        url = create_url(
            # symbol=self.data_config['symbol'],
            symbol=symbol,
            range_=self.data_config['range'],
            datatype=self.data_config['datatype'],
            token=self.data_config['token'])
        response = requests.get(url)
        if "Error Message" in response.text:
            # print('skipping', self.data_config['symbol'], '(not available)')
            return pd.DataFrame()
        data = pd.read_csv(StringIO(response.text))
        data = data[['date', 'open', 'high', 'low', 'close', 'volume']]
        data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        data['timestamp'] = data['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').date())
        data.sort_values(by='timestamp', ascending=True, inplace=True)
        return data

    def load_local_data(self, symbol) -> pd.DataFrame:
        """
        :return: locally saved csv files (yahoo.finance)
        """
        files = os.listdir(self.data_config['source_path'])
        files_with_symbol = [f for f in files if symbol in f]
        if not files_with_symbol:
            print(f"no file with symbol {symbol} in local data - make sure to download file first")
            return
        file = sorted(files_with_symbol)[-1]
        data = pd.read_csv(self.data_config['source_path'] + file)
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if 'adjusted_close' in data.columns:
            columns.append('adjusted_close')
        if 'split_coefficient' in data.columns:
            columns.append('split_coefficient')
        data = data[columns]
        data['timestamp'] = data['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').date())
        data.sort_values(by='timestamp', ascending=True, inplace=True)
        return data
