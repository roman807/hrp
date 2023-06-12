import numpy as np
import pandas as pd
import datetime

from utils.utils import get_config
from anlys.anlys_utils.plot_prices import get_plot
from anlys.anlys_utils.plot_table import df2dash_table
from utils.parsers import anlys_parser
from utils.data_loader import DataLoader
from utils.market_data import MarketData

import dash
from dash import html
from dash import dcc

YRS_LOOK_BACK_PLOT = 1
MAX_YRS_LOOK_BACK = 5


def get_trend(df_prices, n_days, margin=1.05):
    moving_avg = df_prices.rolling(n_days).mean().iloc[-1, :].tolist()
    current = df_prices.iloc[-1, :].tolist()
    trend = []
    for c, m in zip(current, moving_avg):
        if c > margin * m:
            trend.append('up')
        elif c < (1 / margin) * m:
            trend.append('down')
        else:
            trend.append('flat')
    return trend


def main():
    # get user inputs
    parser = anlys_parser()
    args = parser.parse_args()
    data_conf = get_config(args.data_conf)

    returns_to_date = datetime.date.today()
    returns_from_date = returns_to_date - datetime.timedelta(days=365*MAX_YRS_LOOK_BACK)
    data_loader = DataLoader(data_conf, returns_from_date, returns_to_date)
    data_loader.load_data()
    data_loader.calculate_prices_and_returns()
    market_data = MarketData(data_loader.df_returns)

    res = {}
    df_prices, df_returns = data_loader.df_prices, data_loader.df_returns
    first_date, last_date = df_prices.index.min(), df_prices.index.max()

    res['ticker'] = market_data.universe
    res['last close'] = df_prices.loc[last_date, :].values
    res['1y min'] = df_prices[-252:-1].min()
    res['1y max'] = df_prices[-252:-1].max()
    res['price in range [%]'] = np.round((res['last close']-res['1y min']) / (res['1y max']-res['1y min']) * 100, 2)
    res['1day return'] = np.round(df_returns.loc[last_date, :], 3)
    res['1w return'] = np.round(df_prices.iloc[-1, :] / df_prices.iloc[-5, :] - 1, 3)
    res['1m return'] = np.round(df_prices.iloc[-1, :] / df_prices.iloc[-21, :] - 1, 3)
    res['1y return'] = np.round(df_prices.iloc[-1, :] / df_prices.iloc[-252, :] - 1, 3)
    res['1y var'] = np.round(np.diag(market_data.cov_mat) * 252, 3)
    res['15d trend'] = get_trend(df_prices, n_days=15)
    res['50d trend'] = get_trend(df_prices, n_days=50)
    res['200d trend'] = get_trend(df_prices, n_days=200)
    df_res = pd.DataFrame(res)

    df_plot = data_loader.df_prices.iloc[-252*YRS_LOOK_BACK_PLOT:-1, :]
    df_prizes_normalized = df_plot / df_plot.iloc[0, :]

    app = dash.Dash()
    app.layout = html.Div([
        html.Label(
            'Tickers and key indicators as of {}'.format(last_date),
            style=dict(fontSize=16, fontFamily="Arial")#, fontWeight="bold")
        ),
        html.Td(),
        df2dash_table(df_res),
        dcc.Graph(id='graph', figure=get_plot(df_prizes_normalized))
        ],
        style={'margin-top': '3vh', 'margin-left': '3vh'}
    )
    app.run_server(debug=True, port=8011, host='localhost')


if __name__ == '__main__':
    main()
