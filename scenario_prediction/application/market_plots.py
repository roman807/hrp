import plotly.express as px
from datetime import timedelta
import numpy as np
import pandas as pd
import dash
from dash import html
from dash import dcc

from scenario_prediction.application.sp_utils import scale_dataset

_features_price_plot = [
    'moving_avg_5',
    'moving_avg_20',
    'moving_avg_251',
    'price',
]

_features_vola_plot = [
    'vola_over_5',
    'vola_over_20',
    'vola_over_251',
]

_features_histograms = [
    'current_over_moving_avg_5',
    'current_over_moving_avg_20',
    'current_over_moving_avg_251',
    'vola_over_5',
    'vola_over_20',
    'vola_over_251',
]


class MarketPlots:
    def __init__(self, all_datasets, all_predictions, latest_predictions, years_lookback):
        self.market_dataset = all_datasets['SPY']
        self.all_predictions = all_predictions
        self.latest_predictions = latest_predictions
        self.years_lookback = years_lookback
        self.min_date_for_line_plots = self.market_dataset.index[-1] - timedelta(days=years_lookback * 251)
        self.max_date = self.market_dataset.index[-1]

    def get_plot_prices(self):
        df = self.market_dataset[_features_price_plot][self.market_dataset.index >= self.min_date_for_line_plots]
        df.columns.name = 'indicator'
        fig = px.line(df, color_discrete_sequence=['lightgrey', 'darkgrey', 'grey', 'blue'], )
        fig.update_layout(
            height=400,
            width=1350,
            title_text=f"SPY price and moving averages from {self.min_date_for_line_plots.date()} to {self.max_date.date()}",
            xaxis={'title': ""},
            yaxis={'title': "price"}
        )
        return fig

    def get_plot_vola(self):
        df = self.market_dataset[_features_vola_plot][self.market_dataset.index >= self.min_date_for_line_plots]
        df.columns.name = 'indicator'
        fig = px.line(df, color_discrete_sequence=['lightgrey', 'darkgrey', 'grey'], )
        fig.update_layout(
            height=300,
            width=1350,
            title_text=f"SPY volatility from {self.min_date_for_line_plots.date()} to {self.max_date.date()}",
            xaxis={'title': ""},
            yaxis={'title': "volatility"}
        )
        return fig

    def get_histograms(self):
        df = self.market_dataset[_features_histograms]
        df = scale_dataset(df, _features_histograms, scaler='StandardScaler')
        df = df.loc[df[(df.max(axis=1) < 5) & (df.min(axis=1) > -5)].index, :]
        df.columns.name = 'indicator'
        fig = px.histogram(df, facet_col="indicator", facet_col_wrap=6, nbins=80, histnorm='probability density',
                           orientation='h',
                           labels={'current_over_moving_avg_20': 'Price / 20-day moving average'},
                           color_discrete_sequence=['blue', 'blue', 'blue', 'red', 'red', 'red'],
                           opacity=0.6
                           )
        for a in fig.layout.annotations:
            a.text = a.text.split("=")[1]
        for i, feature in enumerate(_features_histograms):
            fig.add_hline(y=np.median(df[feature]), line_dash='dash', line_color='black', col=i+1,
                          row=1)
            fig.add_hline(y=df[feature].values[-1], line_color='firebrick',
                          label={'text': 'current', 'textposition': 'end',
                                 'font': {'size': 10, 'color': 'firebrick', }},
                          line_width=3, col=i+1, row=1)

        fig.update_layout(
            height=300,
            width=1300,
            title_text=f"Indicator-histograms current vs. history since {df.index[0].date()} (normalized values)",
            showlegend=False,
        )
        fig.update_xaxes(visible=False, showticklabels=False)
        fig.update_yaxes(visible=False, showticklabels=False)
        return fig

    def get_prediction_plot(self):

        df = pd.DataFrame(self.latest_predictions, index=['down', 'flat', 'up']).transpose()
        df = df[['up', 'down']]
        df.loc[:, 'down'] = df.loc[:, 'down'].apply(lambda x: -x)
        df.columns.name = "expectation"
        df.index = ['1mo', '2mo', '3mo']

        percentile_for_thresholds = 75
        high_opportunity = 0
        high_risk = 0
        for key in self.all_predictions.keys():
            high_opportunity = max(high_opportunity, np.percentile(self.all_predictions[key][:, 2], percentile_for_thresholds))
            high_risk = max(high_risk, np.percentile(self.all_predictions[key][:, 0], percentile_for_thresholds))

        fig = px.bar(df, color_discrete_sequence=['limegreen', 'red'])
        fig.add_hline(y=high_risk,
                      line_color='darkblue',
                      label={'text': 'high opportunity', 'textposition': 'end',
                              'font': {'size': 10, 'color': 'darkblue', }},
                      line_width=3)
        fig.add_hline(y=-high_risk,
                      line_color='firebrick',
                      label={'text': 'high risk', 'textposition': 'end', 'font': {'size': 10, 'color': 'firebrick', }},
                      line_width=3)
        fig.update_layout(
            height=350,
            width=450,
            title_text=f"SPY predictions as of {self.max_date.date()}",
            xaxis={'title': ""},
            yaxis={'title': "probability"},
            yaxis_range=[-.55, .55]
        )
        return fig

    def run_dashboard(self):
        app = dash.Dash()

        app.layout = html.Div([
            html.Div(
                [dcc.Graph(id='graph1', figure=self.get_plot_prices())],
                style={'margin-top': '0px', 'margin-bottom': '-45px'}
            ),
            html.Div(
                [dcc.Graph(id='graph2', figure=self.get_plot_vola())],
                style={'margin-top': '0px', 'margin-bottom': '-45px'}
            ),
            html.Div(
                [dcc.Graph(id='graph', figure=self.get_histograms())],
                style={'margin-top': '0px', 'margin-bottom': '-45px'}
            ),
            html.Div(
                [dcc.Graph(id='graph3', figure=self.get_prediction_plot())],
                style={'margin-top': '0px', 'margin-bottom': '-45px'}
            ),
        ],
            style={'margin-top': '3vh', 'margin-left': '3vh'}
        )

        app.run_server(debug=True, port=8013, host='localhost')