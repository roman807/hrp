import requests
import numpy as np
import pandas as pd
import os
import re
from io import StringIO
from datetime import datetime

METHODS = {
    'alphavantage': 'load_data_alphavantage',
    'iex': 'load_data_iex',
    'local_sample_data': 'load_sample_data'
}


class DataLoader:
    def __init__(self, data_config: dict):
        self.data_config = data_config
        self.data = self.load_data()

    def load_data(self):
        return getattr(DataLoader, METHODS[self.data_config['api']])(self)

    def load_data_alphavantage(self) -> pd.DataFrame:
        """
        :return: daily prices from alphavantage: https://www.alphavantage.co/
        """
        def create_url(function, symbol, apikey, outputsize, datatype):
            return 'https://www.alphavantage.co/query?function=' + function + '&symbol=' + symbol +'&apikey=' + apikey + \
                   '&outputsize=' + outputsize + '&datatype=' + datatype

        url = create_url(
            function=self.data_config['function'],
            symbol=self.data_config['symbol'],
            apikey=self.data_config['apikey'],
            outputsize=self.data_config['outputsize'],
            datatype=self.data_config['datatype'])
        response = requests.get(url)
        if "Error Message" in response.text:
            # print('skipping', self.data_config['symbol'], '(not available)')
            return pd.DataFrame()
        data = pd.read_csv(StringIO(response.text))
        data['timestamp'] = data['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').date())
        data.sort_values(by='timestamp', ascending=True, inplace=True)
        return data

    def load_data_iex(self) -> pd.DataFrame:
        """
        :return: daily prices from https://iexcloud.io/
        """
        def create_url(symbol, range_, datatype, token):
            return 'https://cloud.iexapis.com/stable/stock/' + symbol + '/chart/' + range_ + '?format=' + datatype + \
                   '&token=' + token
        url = create_url(
            symbol=self.data_config['symbol'],
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

    def load_sample_data(self) -> pd.DataFrame:
        """
        :return: locally saved csv files (yahoo.finance)
        """
        files = os.listdir(self.data_config['path'])
        file = [f for f in files if self.data_config['symbol'] in f][0]
        data = pd.read_csv(self.data_config['path'] + file)
        data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        # data['timestamp'] = data['timestamp'].apply(lambda x: x.replace('.', '-'))   # remove if csv timestamp in proper format
        # data['timestamp'] = data['timestamp'].apply(lambda x: datetime.strptime(x, '%d-%m-%y').date())
        data['timestamp'] = data['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').date())
        data.sort_values(by='timestamp', ascending=True, inplace=True)
        return data

    def get_returns(self, min_date):
        data = self.data.copy()
        data['returns'] = [0.0] + (np.log(np.array(data['close'])[1:] / np.array(data['close'])[:-1])).tolist()
        ts = data[data['timestamp'] >= min_date][['timestamp', 'returns']]
        ts = ts.set_index('timestamp')
        ts.columns = [self.data_config['symbol']]
        return ts




