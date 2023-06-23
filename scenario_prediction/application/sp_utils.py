from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def print_data_stats(all_datasets, ml_conf):
    print(f'\nstats for data with margin {ml_conf["data"]["margin"]} and target in {ml_conf["data"]["n_future_days_target"]} future days:')
    for symbol in all_datasets.keys():
        n_rows = all_datasets[symbol].shape[0]
        up = sum(all_datasets[symbol]['target'] == 'up')
        down = sum(all_datasets[symbol]['target'] == 'down')
        flat = sum(all_datasets[symbol]['target'] == 'flat')
        print(
            f' {symbol} - up: {round(100 * up / n_rows)}%, flat: {round(100 * flat / n_rows)}%, down: {round(100 * down / n_rows)}%'
        )


def scale_datasets(df_train, df_test, features, scaler='StandardScaler'):
    if scaler == 'StandardScaler':
        scaler_ = StandardScaler()
    elif scaler == 'MinMaxScaler':
        scaler_ = MinMaxScaler()
    else:
        raise Exception(f"scaler '{scaler}' not available")
    scaler_.fit(df_train[features])
    train_features_transformed = scaler_.transform(df_train[features])
    df_train_scaled = pd.DataFrame(train_features_transformed, index=df_train.index, columns=features)
    df_train_scaled = pd.merge(df_train['target'], df_train_scaled, left_index=True, right_index=True)
    if df_test.shape[0] > 0:
        test_features_transformed = scaler_.transform(df_test[features])
        df_test_scaled = pd.DataFrame(test_features_transformed, index=df_test.index, columns=features)
        df_test_scaled = pd.merge(df_test['target'], df_test_scaled, left_index=True, right_index=True)
    else:
        df_test_scaled = df_test
    return df_train_scaled, df_test_scaled


def scale_dataset(df, features, scaler='StandardScaler'):
    if scaler == 'StandardScaler':
        scaler_ = StandardScaler()
    elif scaler == 'MinMaxScaler':
        scaler_ = MinMaxScaler()
    else:
        raise Exception(f"scaler '{scaler}' not available")
    scaler_.fit(df[features])
    features_transformed = scaler_.transform(df[features])
    df_scaled = pd.DataFrame(features_transformed, index=df.index, columns=features)
    if 'target' in df.columns:
        df_scaled = pd.merge(df['target'], df_scaled, left_index=True, right_index=True)
    return df_scaled


def get_train_and_test_sets(all_datasets, ml_conf):
    test_sets = [
        (datetime.strptime('2007-01-01', '%Y-%m-%d'), datetime.strptime('2010-12-31', '%Y-%m-%d')),
        (datetime.strptime('2011-01-01', '%Y-%m-%d'), datetime.strptime('2014-12-31', '%Y-%m-%d')),
        (datetime.strptime('2015-01-01', '%Y-%m-%d'), datetime.strptime('2018-12-31', '%Y-%m-%d')),
        (datetime.strptime('2019-01-01', '%Y-%m-%d'), datetime.strptime('2022-12-31', '%Y-%m-%d')),
    ]
    all_train_test_set_combinations = []
    for test_set in test_sets:
        all_train_sets = []
        all_test_sets = []
        for symbol in eval(ml_conf['data']['symbols_to_include']):
            df_test = all_datasets[symbol][
                (all_datasets[symbol].index >= test_set[0]) & (all_datasets[symbol].index <= test_set[1])
                ]
            # all_test_sets.append(df_test)
            if df_test.shape[0] > 0:
                pos_start = all_datasets[symbol].index.get_loc(df_test.index[0])
                pos_end = all_datasets[symbol].index.get_loc(df_test.index[-1])
                df_train1 = all_datasets[symbol].iloc[:pos_start - ml_conf['data']['n_future_days_target'], :]
                df_train2 = all_datasets[symbol].iloc[pos_end + ml_conf['data']['n_future_days_target']:, :]
                df_train = pd.concat([df_train1, df_train2], axis=0)
            else:
                df_train = all_datasets[symbol]
            if eval(ml_conf['data']['scale_datasets']):
                df_train, df_test = scale_datasets(df_train, df_test, ml_conf['features'],
                                                   ml_conf['data']['scaler'])
            all_test_sets.append(df_test)
            all_train_sets.append(df_train)

        df_train_combined = pd.concat(all_train_sets)
        df_test_combined = pd.concat(all_test_sets)
        all_train_test_set_combinations.append((df_train_combined, df_test_combined))
    return all_train_test_set_combinations


def get_train_set_for_inference(all_datasets, ml_conf):
    all_train_sets = []
    for symbol in eval(ml_conf['data']['symbols_to_include']):
        df_train = all_datasets[symbol]
        if eval(ml_conf['data']['scale_datasets']):
            df_train = scale_dataset(df_train, ml_conf['features'], ml_conf['data']['scaler'])
        all_train_sets.append(df_train)

    return pd.concat(all_train_sets)


def get_data_for_inference(all_datasets, ml_conf):
    df = all_datasets[ml_conf['data']['symbol_for_inference']][ml_conf['features']]
    if eval(ml_conf['data']['scale_datasets']):
        df = scale_dataset(df, ml_conf['features'], ml_conf['data']['scaler'])
    return df# pd.DataFrame(df.iloc[-1, :])#.reshape(-1, 1)
