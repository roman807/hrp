# Idea: use current price vs. 3-, 5-, 15-, 20- etc. -day moving averages to predict if average price over the next
# month (i.e. 21-day fwd moving average) will be higher than current price. If so, this would indicate a good time to
# buy (assuming the stock is expected to perform well in the long term)

# Tried simple strategies (e.g. buy when 5-day moving average below 2%) and ML classifiers with different moving
# averages as input features. Used daily price-data of all available S&P500 tickers from 2010-01 to 2023-06. Results
# were not convincing.

# Other things to try next: normalize stock prices by S&P, sota indicators, trend-following strategies


import datetime
from time import time
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from utils.data_loader import DataLoader
from utils.parsers import anlys_parser
from utils.utils import get_config


MAX_YRS_LOOK_BACK = 5
MOVING_AVERAGE_NUMBER_OF_DAYS = [3, 5, 10, 15, 30]
MARGIN_FOR_STRATEGY = 0.015
MARGIN_CONSIDERED_FLAT = 0.005
TEST_SIMPLE_STRATEGY_YEAR_BY_YEAR = False


def eval_predictions(model_name, y_true, y_pred):
    precision = round(precision_score(y_true, y_pred), 2)
    precision_rnd = round(y_true.sum() / len(y_true), 2)
    percentage_action = round(100 * y_pred.sum() / len(y_pred), 1)
    print(
        f'{model_name} precision: {precision} (random guess: {precision_rnd}, percentage of action: {percentage_action}%)'
    )


def main():
    parser = anlys_parser()
    args = parser.parse_args()
    data_conf = get_config(args.data_conf)

    returns_to_date = datetime.date.today()
    returns_from_date = returns_to_date - datetime.timedelta(days=365 * MAX_YRS_LOOK_BACK)
    data_loader = DataLoader(data_conf, returns_from_date, returns_to_date)
    data_loader.load_data()
    datasets = data_loader.get_datasets_for_mean_reversion(MOVING_AVERAGE_NUMBER_OF_DAYS)
    print(f'\nnumber of datasets considered: {len(datasets)}')
    df_concat = pd.concat([v for _, v in datasets.items()])
    print(f'total number of rows: {df_concat.shape[0]}')

    df_concat = pd.concat([v for _, v in datasets.items()])
    df_concat['buy'] = df_concat['fwd'] >= MARGIN_CONSIDERED_FLAT
    df_concat['sell'] = df_concat['fwd'] <= -MARGIN_CONSIDERED_FLAT
    df_concat[['buy', 'sell']] = df_concat[['buy', 'sell']].astype('int')

    print('\n********** Simple strategies **********')
    print(f'\nnumber of datasets considered: {len(datasets)}')
    print(f'total number of rows: {df_concat.shape[0]}')

    print('\nBuy-precision scores')
    for n in MOVING_AVERAGE_NUMBER_OF_DAYS:
        pred_buy = (df_concat[f'moving_avg_{n}'] <= -MARGIN_FOR_STRATEGY).astype(int)
        precision = round(precision_score(df_concat["buy"], pred_buy), 2)
        precision_rnd = round(df_concat["buy"].sum() / df_concat.shape[0], 2)
        percentage_buy = round(100 * pred_buy.sum() / df_concat.shape[0], 1)
        print(f'If buy when {n}-day-moving-avg below -{round(100*MARGIN_FOR_STRATEGY, 1)}%: {precision} (random guess: {precision_rnd}, percentage of buys: {percentage_buy}%)')

    print('\nSell-precision scores')
    for n in MOVING_AVERAGE_NUMBER_OF_DAYS:
        pred_sell = (df_concat[f'moving_avg_{n}'] >= MARGIN_FOR_STRATEGY).astype(int)
        precision = round(precision_score(df_concat["sell"], pred_sell), 2)
        precision_rnd = round(df_concat["sell"].sum() / df_concat.shape[0], 2)
        percentage_sell = round(100 * pred_sell.sum() / df_concat.shape[0], 1)
        print(f'If sell when {n}-day-moving-avg below -{round(100 * MARGIN_FOR_STRATEGY, 1)}%: {precision} (random guess: {precision_rnd}, percentage of buys: {percentage_sell}%)')

    # check year by year
    if TEST_SIMPLE_STRATEGY_YEAR_BY_YEAR:
        print('\n********** precision scores year-by-year **********')
        df_concat['date'] = df_concat.index
        df_concat['date'] = df_concat['date'].apply(lambda x: pd.to_datetime(x))
        df_concat['year'] = df_concat['date'].apply(lambda x: x.year)
        for n in MOVING_AVERAGE_NUMBER_OF_DAYS:
            print(f'\n if buy when {n}-day-moving-avg below -{round(100 * MARGIN_FOR_STRATEGY, 1)}%')
            for year in sorted(df_concat['year'].unique()):
                df_year = df_concat[df_concat['year'] == year]
                pred_buy = (df_year[f'moving_avg_{n}'] <= -MARGIN_FOR_STRATEGY).astype(int)
                precision = round(precision_score(df_year["buy"], pred_buy), 2)
                precision_rnd = round(df_year["buy"].sum() / df_year.shape[0], 2)
                print(f'{year} - {precision} ({precision_rnd})')

    print('\n********** ML models **********')
    features = [f'moving_avg_{n}' for n in MOVING_AVERAGE_NUMBER_OF_DAYS]

    # prepare data
    RANDOM_STATE = 1
    TEST_SIZE = 0.33
    X = df_concat[features]
    X_train, X_test, y_train_buy, y_test_buy = train_test_split(
        X, df_concat['buy'], test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    _, _, y_train_sell, y_test_sell = train_test_split(
        X, df_concat['sell'], test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f'number of train samples: {X_train.shape[0]} ({round((1-TEST_SIZE)*100)}%)')
    print(f'number of test samples: {X_test.shape[0]} ({round(TEST_SIZE*100)}%)')
    print(f'features: {features}')

    print('\n********** logistic regression **********')
    # Logistic Regression - buy model
    clf = LogisticRegression(class_weight='balanced')
    start = time()
    clf.fit(X_train, y_train_buy)
    print(f'trained LR-buy in {round(time()-start)} seconds')
    pred = clf.predict(X_test)
    eval_predictions('LR-buy', y_test_buy, pred)

    # Logistic Regression - sell model
    clf = LogisticRegression(class_weight='balanced')
    start = time()
    clf.fit(X_train, y_train_sell)
    print(f'trained LR-sell in {round(time()-start)} seconds')
    pred = clf.predict(X_test)
    eval_predictions('LR-sell', y_test_sell, pred)

    print('\n********** random forests **********')
    # Random Forest - buy model
    # clf = RandomForestClassifier()
    clf = RandomForestClassifier(n_estimators=25, max_depth=5, random_state=RANDOM_STATE)
    start = time()
    clf.fit(X_train, y_train_buy)
    print(f'trained RF-buy in {round(time()-start)} seconds')
    pred = clf.predict(X_test)
    eval_predictions('RF-buy', y_test_buy, pred)

    # Random Forest - sell model
    clf = RandomForestClassifier(n_estimators=25, max_depth=5, random_state=RANDOM_STATE)
    start = time()
    clf.fit(X_train, y_train_sell)
    print(f'trained RF-sell in {round(time()-start)} seconds')
    pred = clf.predict(X_test)
    eval_predictions('RF-sell', y_test_sell, pred)

    print('\n********** GBM **********')
    # GBM - buy model
    clf = GradientBoostingClassifier(n_estimators=25, max_depth=5, random_state=RANDOM_STATE)
    start = time()
    clf.fit(X_train, y_train_buy)
    print(f'trained GBM-buy in {round(time()-start)} seconds')
    pred = clf.predict(X_test)
    eval_predictions('GBM-buy', y_test_buy, pred)

    # GBM - sell model
    clf = GradientBoostingClassifier(n_estimators=25, max_depth=5, random_state=RANDOM_STATE)
    start = time()
    clf.fit(X_train, y_train_sell)
    print(f'trained GBM-sell in {round(time()-start)} seconds')
    pred = clf.predict(X_test)
    eval_predictions('GBM-sell', y_test_sell, pred)


if __name__ == '__main__':
    main()
