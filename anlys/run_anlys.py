import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time

from utils.utils import get_config
from anlys.anlys_utils.plot_prices import get_plot
from anlys.anlys_utils.plot_table import df2dash_table
from anlys.anlys_utils.anlys_utils import anlys_parser
from utils.data_loader import DataLoader
from utils.market_data import MarketData

import dash
import dash_html_components as html
import dash_core_components as dcc

YRS_LOOK_BACK = 1


def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


def main():
    start_time = time.time()
    # get user inputs
    parser = anlys_parser()
    args = parser.parse_args()
    data_conf = get_config(args.data_conf)

    returns_to_date = datetime.date.today()
    returns_from_date = returns_to_date - datetime.timedelta(days=365*YRS_LOOK_BACK)
    data_loader = DataLoader(data_conf, returns_from_date, returns_to_date)
    data_loader.load_data()
    market_data = MarketData(data_loader.df_returns)

    res = {}
    df_prices, df_returns = data_loader.df_prices, data_loader.df_returns
    first_date, last_date = df_prices.index.min(), df_prices.index.max()
    # last_date = data_loader.df_prices.index.max()
    res['ticker'] = market_data.universe
    res['last close'] = df_prices.loc[last_date, :].values
    res['1yr min'] = df_prices.min()
    res['1yr max'] = df_prices.max()
    res['price in range [%]'] = np.round((res['last close']-res['1yr min']) / (res['1yr max']-res['1yr min']) * 100, 2)
    res['1day return'] = np.round(df_returns.loc[last_date, :], 3)
    res['1week return'] = np.round(df_prices.iloc[-1, :] / df_prices.iloc[-5, :] - 1, 3)
    res['1month return'] = np.round(df_prices.iloc[-1, :] / df_prices.iloc[-21, :] - 1, 3)
    res['1yr return'] = np.round(df_prices.iloc[-1, :] / df_prices.iloc[0, :] - 1, 3)
    res['1yr variance'] = np.round(np.diag(market_data.cov_mat) * 252, 3)
    df_res = pd.DataFrame(res)
    print(df_res)

    df_prizes_normalized = data_loader.df_prices / data_loader.df_prices.iloc[0, :]

    app = dash.Dash()
    app.layout = html.Div([
        html.Label(
            'Tickers and key indicators as of {}'.format(last_date),
            style=dict(fontSize=16, fontFamily="Arial")#, fontWeight="bold")
        ),
        df2dash_table(df_res),
        dcc.Graph(id='graph', figure=get_plot(df_prizes_normalized))
    ])
    app.run_server(debug=True, port=8010, host='localhost')


if __name__ == '__main__':
    main()
