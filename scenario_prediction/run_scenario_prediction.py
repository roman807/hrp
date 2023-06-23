from utils.data_loader import DataLoader
from utils.utils import get_config
from scenario_prediction.application.sp_utils import (
    print_data_stats,
    get_train_and_test_sets,
    get_train_set_for_inference,
    get_data_for_inference,
)
from scenario_prediction.application.ml_trainer import MLTrainer
from scenario_prediction.application.market_plots import MarketPlots


# ---------- input configuration ---------- #

# specify configs for ML models
_ml_conf_file_1mo = 'configs/scenario_prediction/logistic_regression_1mo.json'
_ml_conf_file_2mo = 'configs/scenario_prediction/logistic_regression_2mo.json'
_ml_conf_file_3mo = 'configs/scenario_prediction/logistic_regression_3mo.json'

# _data_conf_file = 'configs/scenario_prediction/dataconf_markets_alphavantage.json'
_data_conf_file = 'configs/scenario_prediction/dataconf_markets_local.json'

_model_dir = 'scenario_prediction/models'

_run_performance_validation = True
_train_new_models = True
_run_dashboard = True
_years_loockback_for_plots = 5


def main():
    ml_confs = [
        get_config(_ml_conf_file_1mo),
        get_config(_ml_conf_file_2mo),
        get_config(_ml_conf_file_3mo),
    ]
    data_conf = get_config(_data_conf_file)

    # ---------- step 1: create raw datasets ---------- #
    data_loader = DataLoader(data_conf)
    save_as_csv = True if 'local' not in _data_conf_file else False
    data_loader.load_data(save_as_csv=save_as_csv)

    # ---------- step 2: train models for performance validation ---------- #
    if _run_performance_validation:
        for ml_conf in ml_confs:

            # step 2a: prepare train and test data:
            all_datasets = data_loader.get_datasets_for_scenario_prediction(
                ml_conf['data']['margin'],
                ml_conf['data']['n_future_days_target']
            )
            print_data_stats(all_datasets, ml_conf)
            all_train_test_set_combinations = get_train_and_test_sets(all_datasets, ml_conf)

            # step 2b: train models
            ml_trainer = MLTrainer(ml_conf)
            ml_trainer.train_for_performance_validation(all_train_test_set_combinations)

    # ---------- step 3: train models for inference ---------- #
    if _train_new_models:
        for ml_conf in ml_confs:
            all_datasets = data_loader.get_datasets_for_scenario_prediction(
                ml_conf['data']['margin'],
                ml_conf['data']['n_future_days_target'],
            )
            train_set = get_train_set_for_inference(all_datasets, ml_conf)
            ml_trainer = MLTrainer(ml_conf)
            ml_trainer.train_for_inference(train_set)
            ml_trainer.save_model(_model_dir)

    # ---------- step 4: perform inference on latest data point ---------- #
    all_predictions = {}
    latest_predictions = {}
    all_datasets = data_loader.get_datasets_for_scenario_prediction(with_target=False)
    for ml_conf in ml_confs:
        ml_trainer = MLTrainer(ml_conf)
        ml_trainer.load_model(_model_dir)
        data_inference = get_data_for_inference(all_datasets, ml_conf)
        predictions = ml_trainer.clf.predict_proba(data_inference)
        all_predictions[ml_conf['model']['name']] = predictions
        latest_predictions[ml_conf['model']['name']] = predictions[-1]

    # ---------- step 5: run dashboard ---------- #
    if _run_dashboard:
        market_plots = MarketPlots(all_datasets, all_predictions, latest_predictions, _years_loockback_for_plots)
        market_plots.run_dashboard()


if __name__ == '__main__':
    main()
