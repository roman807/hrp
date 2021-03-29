import dash_table


def df2dash_table(df):
    return dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i}
                 for i in df.columns],
        data=df.to_dict('records'),
        style_cell=dict(textAlign='left', minWidth="80px"),
        style_header=dict(backgroundColor="lightgrey", fontSize=14, fontFamily="Arial"),
        style_data=dict(backgroundColor="white"),
        fill_width=False,
        # margin=dict(
        #     l=100
        # ),
    )