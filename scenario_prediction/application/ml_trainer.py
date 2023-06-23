import numpy as np
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
import pickle
import os


class MLTrainer:
    def __init__(self, ml_config):
        self.ml_config = ml_config
        self.clf = None

    def get_clf(self):
        params = self.ml_config['model']['params']
        algo = self.ml_config['model']['algo']

        if algo == "LogisticRegression":
            return LogisticRegression(
                multi_class=params['multi_class'],
                class_weight=params['class_weight']
            )
        if algo == "RandomForestClassifier":
            return RandomForestClassifier(
                n_estimators=params['n_estimators'],
                criterion=params['criterion'],
                max_depth=params['max_depth'],
                max_features=params['max_features'],
                class_weight=params['class_weight'],
            )
        if algo == "GradientBoostingClassifier":
            return GradientBoostingClassifier(
                loss=params['loss'],
                learning_rate=params['learning_rate'],
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                max_features=params['max_features'],
            )

    def print_performance(self, true, pred, setting_desc):
        random_pred = true.copy()
        np.random.shuffle(random_pred)
        f1 = round(f1_score(true, pred, average="weighted"), 3)
        f1_random = round(f1_score(true, random_pred, average="weighted"), 3)
        accuracy = round(100*accuracy_score(true, pred),1)
        accuracy_random = round(100 * accuracy_score(true, random_pred), 1)
        up_pred = round(100*sum(pred=="up") / len(pred))
        flat_pred = round(100 * sum(pred == "flat") / len(pred))
        down_pred = round(100 * sum(pred == "down") / len(pred))
        up_true = round(100*sum(true=="up")/len(true))
        flat_true = round(100 * sum(true == "flat") / len(true))
        down_true = round(100 * sum(true == "down") / len(true))
        print(f'\nperformance {setting_desc} (predict {self.ml_config["data"]["n_future_days_target"]} days into future):')
        print(f'F1-score {f1} (random: {f1_random})')
        print(f'accuracy {accuracy}% (random: {accuracy_random}%)')
        print(f' up - pred: {up_pred}% (true: {up_true}%)')
        print(f' flat - pred: {flat_pred}% (true: {flat_true}%)')
        print(f' down - pred: {down_pred}% (true: {down_true}%)')

    def train_for_performance_validation(
            self,
            all_train_test_set_combinations,
            print_train_performance=False,
            print_test_fold_performance=False,
    ):
        all_y_train_pred = np.array([])
        all_y_test_pred = np.array([])
        all_y_train_true = np.array([])
        all_y_test_true = np.array([])

        start = time()
        for i, (df_train, df_test) in enumerate(all_train_test_set_combinations):
            X_train = df_train[self.ml_config['features']]
            y_train = df_train['target']
            X_test = df_test[self.ml_config['features']]
            y_test = df_test['target']

            clf = self.get_clf()
            clf.fit(X_train, y_train)
            train_pred = clf.predict(X_train)
            all_y_train_pred = np.concatenate([all_y_train_pred, train_pred])
            all_y_train_true = np.concatenate([all_y_train_true, y_train.to_numpy()])
            if print_train_performance and print_test_fold_performance:
                self.print_performance(y_train, train_pred, f'training fold {i}')

            test_pred = clf.predict(X_test)
            all_y_test_pred = np.concatenate([all_y_test_pred, test_pred])
            all_y_test_true = np.concatenate([all_y_test_true, y_test.to_numpy()])
            if print_test_fold_performance:
                self.print_performance(y_test, test_pred, f'test fold {i}')

        print(f'\n\n ********** training complete - total training time: {round(time() - start)} seconds **********')

        print('\nparameters:')
        print(f'total training time: {round(time() - start)} seconds')
        print(f' margin: {round(self.ml_config["data"]["margin"] * 100, 1)}%')
        print(f' n_future_days_target: {self.ml_config["data"]["n_future_days_target"]}')
        print(f' symbols included: {self.ml_config["data"]["symbols_to_include"]}')
        print(f' features used: {self.ml_config["features"]}')

        if print_train_performance:
            self.print_performance(all_y_train_true, all_y_train_pred, f'train combined')
        self.print_performance(all_y_test_true, all_y_test_pred, f'test combined')

    def train_for_inference(self, df_train):
        X_train = df_train[self.ml_config['features']]
        y_train = df_train['target']
        self.clf = self.get_clf()
        self.clf.fit(X_train, y_train)

    def save_model(self, model_dir):
        file_name = f'{model_dir}/{self.ml_config["model"]["name"]}.sav'
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'wb') as f:
            pickle.dump(self.clf, f)
        print(f'\nsaved model as {file_name}')

    def load_model(self, model_dir):
        file_name = f'{model_dir}/{self.ml_config["model"]["name"]}.sav'
        self.clf = pickle.load(open(file_name, 'rb'))
