import dash_table


def df2dash_table(df):
    return dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i}
                 for i in df.columns],
        data=df.to_dict('records'),
        style_cell=dict(textAlign='left', minWidth="80px", height="25px"),
        style_header=dict(backgroundColor="lightgrey", fontSize=13, fontFamily="Arial", fontWeight="bold"),
        style_data=dict(backgroundColor="white"),
        fill_width=False,
        style_as_list_view=True,
        # filter_action="native",
        sort_action="native",
        # style_table=dict(height="500px", overflowY="auto"),
        style_data_conditional=[
            {
                'if': {
                    'filter_query': '{15d trend} = "up"',
                    'column_id': '15d trend'
                },
                'backgroundColor': 'lightgreen'
            },
            {
                'if': {
                    'filter_query': '{15d trend} = "down"',
                    'column_id': '15d trend'
                },
                'backgroundColor': 'lightcoral'
            },
            {
                'if': {
                    'filter_query': '{50d trend} = "up"',
                    'column_id': '50d trend'
                },
                'backgroundColor': 'lightgreen'
            },
            {
                'if': {
                    'filter_query': '{50d trend} = "down"',
                    'column_id': '50d trend'
                },
                'backgroundColor': 'lightcoral'
            },
            {
                'if': {
                    'filter_query': '{200d trend} = "up"',
                    'column_id': '200d trend'
                },
                'backgroundColor': 'lightgreen'
            },
            {
                'if': {
                    'filter_query': '{200d trend} = "down"',
                    'column_id': '200d trend'
                },
                'backgroundColor': 'lightcoral'
            },
        ]
        # margin=dict(
        #     l=100
        # ),
    )