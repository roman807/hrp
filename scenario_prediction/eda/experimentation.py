# run and experiment with this script to find best data configuration (symbols, days fwd, margin, scaling, etc.), best
# algorithm, and parameter-configuration. Best-configurations are then manually saved in configs/scenario_prediction
# and used in "run_scenario_prediction.py"

from scenario_prediction.application.ml_trainer import MLTrainer
from scenario_prediction.application.sp_utils import get_train_and_test_sets, print_data_stats
from utils.data_loader import DataLoader


data_conf = {
    "api": "local_data",
    # "source_path": "data/data_markets_2023-06-15/",
    "source_path": "data/data_markets/",
    "symbols": "['SPY', 'EWJ', 'ASHR', 'EWZ', 'VGK']",
    # "symbols": "['EWJ']",
}

parameters = {
    'data': {
        'margin': 0.03,
        'n_future_days_target': 21,
        'symbols_to_include': "['SPY']",#, 'EWJ', 'ASHR', 'VGK'],
        'scale_datasets': 'False',
        'scaler': 'StandardScaler'
    },
    'features': [
        # 'current_over_moving_avg_5',
        # 'current_over_moving_avg_10',
        # 'current_over_moving_avg_20',
        # 'current_over_moving_avg_63',
        # 'current_over_moving_avg_126',
        'current_over_moving_avg_251',
        # 'return_over_5',
        # 'return_over_10',
        # 'return_over_20',
        # 'return_over_63',
        # 'return_over_126',
        # 'return_over_251',
        'vola_over_5',
        # 'vola_over_10',
        'vola_over_20',
        # 'vola_over_63',
        # 'vola_over_126',
        'vola_over_251',
    ],
    'model': {
        'algo': 'LogisticRegression',
        'params': {
            'multi_class': 'multinomial',
            'class_weight': 'balanced',
        }
    },
    # 'model': {
    #     'algo': 'RandomForestClassifier',
    #     'params': {
    #         'n_estimators': 25,
    #         'criterion': 'gini',
    #         'max_depth': 8,
    #         'max_features': 'sqrt',
    #         'class_weight': 'balanced',
    #     }
    # },
    # 'model': {
    #     'algo': 'GradientBoostingClassifier',
    #     'params': {
    #         'loss': 'log_loss',
    #         'learning_rate': 0.1,
    #         'n_estimators': 20,
    #         'max_depth': 3,
    #         'max_features': 'sqrt',
    #     }
    # },
}

def main():
    # ---------- step 1: create raw datasets ---------- #
    data_loader = DataLoader(data_conf)
    data_loader.load_data(save_as_csv=False)
    all_datasets = data_loader.get_datasets_for_scenario_prediction(
        parameters['data']['margin'],
        parameters['data']['n_future_days_target']
    )
    print_data_stats(all_datasets, parameters)

    # ---------- step 2: create all train & test sets ---------- #
    all_train_test_set_combinations = get_train_and_test_sets(all_datasets, parameters)

    # ---------- step 3: train models ---------- #
    ml_trainer = MLTrainer(parameters)
    ml_trainer.train_for_performance_validation(
        all_train_test_set_combinations,
        print_train_performance=True,
        print_test_fold_performance=False,
    )


if __name__ == '__main__':
    main()
