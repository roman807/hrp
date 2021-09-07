import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def get_plot1(df):
    return px.line(df)


LINE_SIZE = 2
COLORS = [
    'black',
    'darkgrey',
    'lightgrey',
    'steelblue',
    'lightsteelblue',
    'seagreen',
    'mediumseagreen',
    'darkblue',
    'orange',
    'lightgreen',
    'red',
    'darkred',
    'yellow',
    'darkorange',
    'brown',
]


def get_plot(df: pd.DataFrame):
    # reference: https://plotly.com/python/line-charts/

    fig = go.Figure()
    for i in range(df.shape[1]):
        fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, i], mode='lines',
                                 name=df.columns[i],
                                 line=dict(color=COLORS[i], width=LINE_SIZE),
                                 connectgaps=True,
                                 ))
    fig.update_layout(
        width=1100,
        height=500,
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=False,
            showline=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            showticklabels=True,
        ),
        autosize=False,
        margin=dict(
            autoexpand=False,
            l=40,
            r=80,
            t=110,
        ),
        showlegend=True,
        plot_bgcolor='white',
        legend=dict(
            x=1,
            y=1,
            traceorder="reversed",
            title_font_family="Arial",
            font=dict(
                family="Arial",
                size=10,
                color="black"
            )
        )
    )
    annotations = []
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                            xanchor='left', yanchor='bottom',
                            text='Prices since {}'.format(df.index[0]),
                            font=dict(family='Arial',
                                      size=16,
                                      color='rgb(37,37,37)'),
                            showarrow=False))

    fig.update_layout(annotations=annotations)
    return fig
