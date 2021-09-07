import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px


class OptimizationDahsboard:
    def __init__(self, conf, allocations_df, risk_return_df):
        self.conf = conf
        self.allocations_df = allocations_df
        self.risk_return_df = risk_return_df

    def run_app(self):
        app = dash.Dash()

        app.layout = html.Div(className='row', children=[
            html.Label(
                'Optimization result',
                style=dict(fontSize=20, fontFamily="Arial", fontWeight="bold")
            ),
            dcc.Dropdown(
                id="dropdown",
                options=[{"label": "risk appetite: " + str(x), "value": x} for x in self.conf["risk_appetites"]],
                value=self.conf["risk_appetites"][0],
                clearable=False,
                style={'margin-top': '5vh', }
            ),
            html.Div(children=[
                dcc.Graph(id="bar-chart", style={'display': 'inline-block'}),
                dcc.Graph(id="scatter-plot", style={'display': 'inline-block'}),
            ]),
        ],
                              style={'display': 'inline-block'}
                              )

        @app.callback(
            Output("bar-chart", "figure"),
            [Input("dropdown", "value")])
        def update_bar_chart(risk_appetite):
            mask = self.allocations_df["risk_appetite"] == risk_appetite
            fig = px.bar(self.allocations_df[mask], x="weights", y="index", orientation='h', text="weights", barmode="group")
            fig.update_traces(texttemplate='%{text:.1%}', textposition='inside')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            fig.update_layout(title_text='Optimal allocation')
            fig.update_xaxes(visible=False)
            return fig

        @app.callback(
            Output("scatter-plot", "figure"),
            [Input("dropdown", "value")])
        def update_scatter_plot(risk_appetite):
            fig = px.scatter(self.risk_return_df, x="risk", y="expected_return",
                             text="risk_appetite",
                             )
            fig.update_traces(texttemplate='%{text}', textposition='top center',
                              marker_color=['green' if self.risk_return_df['risk_appetite'].values[i] == risk_appetite else 'grey' \
                                            for i in range(self.risk_return_df.shape[0])],
                              marker={'size': [25 if self.risk_return_df['risk_appetite'].values[i] == risk_appetite else 15 \
                                               for i in range(self.risk_return_df.shape[0])]})
            fig.update_xaxes(range=[self.risk_return_df["risk"].min() - .001, self.risk_return_df["risk"].max() + .001])
            fig.update_yaxes(
                range=[self.risk_return_df["expected_return"].min() - .01, self.risk_return_df["expected_return"].max() + .01])
            fig.update_layout(title_text='Risk / return by risk appetite', xaxis_tickformat='.2%', yaxis_tickformat='.2%')
            return fig

        app.run_server(debug=True, port=8012, host='localhost')