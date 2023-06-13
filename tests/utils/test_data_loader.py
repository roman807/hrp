import pandas as pd
import numpy as np
from datetime import datetime

from utils.data_loader import DataLoader

DATA_CONFIG = {
    "api": "local_data",
    "source_path": "tests/utils/",
    "symbols": "['TKR']",
}


def test_load_local_data():
    dataloader = DataLoader(data_config=DATA_CONFIG)
    data = dataloader.load_local_data('TKR')
    assert data['close'].tolist() == [1.0, 1.05, 0.95]


def test_get_returns():
    from_date = datetime.strptime('2000-01-01', '%Y-%m-%d')
    to_date = datetime.strptime('2001-01-01', '%Y-%m-%d')
    data = pd.DataFrame(
        [
            [datetime.strptime('2000-01-01', '%Y-%m-%d'), 1.0],
            [datetime.strptime('2000-01-02', '%Y-%m-%d'), 1.05],
            [datetime.strptime('2000-01-03', '%Y-%m-%d'), 1.1],
        ],
        columns=['timestamp', 'close']
    )
    dataloader = DataLoader(data_config=DATA_CONFIG, from_date=from_date, to_date=to_date)
    returns = dataloader.get_returns(data, 'RND')
    assert returns.iloc[1, 0] == np.log(1.05/1)
