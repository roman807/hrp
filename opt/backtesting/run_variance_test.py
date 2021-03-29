import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os

from opt.optimizers.hierarchical_risk_parity import HierarchicalRiskParity
from opt.optimizers.mean_variance import MeanVariance

from utils.data_loader import DataLoader
from opt.opt_utils.constraints_builder import ConstraintsBuilder
from utils.market_data import MarketData

N_INSTR = 10
N_PORTFOLIOS = 1000
DATE_MIN = pd.to_datetime('2008-01-01')
DATE_MAX = pd.to_datetime('2020-01-01')
DATA_PATH = 'sample_data/s_and_p500/'
CONF = {
  "optimizer": "HierarchicalRiskParity",
  "yrs_look_back": 4,
  "min_weight_constraints": {},
  "max_weight_constraints": {},
  "max_number_instruments": 10,
  "consider_returns": False,
  "risk_appetite": 0.0,
  "expected_returns": {}
}
CONF_DATA = {
    "api": "local_sample_data",
    "function": "TIME_SERIES_DAILY",
    "path": "sample_data/s_and_p500/",
    "symbols": "['AAPL', 'GOOG', 'FB', 'AMZN', 'KO', 'BX']",
    "apikey": "JHC56UFM63VB6RUJ",
    "outputsize": "full",
    "datatype": "csv"
}


def get_random_universe(n_instr):
    return [s.replace('.csv', '') for s in np.random.choice(os.listdir(DATA_PATH), n_instr, replace=False)]


def main():
    # get user inputs
    conf = CONF
    data_conf = CONF_DATA

    np.random.seed(1)
    mv_predicted, mv_real, hrp_predicted, hrp_real = [], [], [], []
    for i in range(N_PORTFOLIOS):
        universe = get_random_universe(N_INSTR)
        ref_date = DATE_MIN + datetime.timedelta(days=int((DATE_MAX - DATE_MIN).days * np.random.random()))
        returns_from_date = ref_date - datetime.timedelta(days=365 * conf['yrs_look_back'])
        returns_to_date = ref_date + datetime.timedelta(days=365)
        data_loader = DataLoader(data_conf, universe, returns_from_date, returns_to_date)
        data_loader.load_data(print_progress=False)

        df_returns_train = data_loader.df_returns[data_loader.df_returns.index <= ref_date]
        df_returns_test = data_loader.df_returns[data_loader.df_returns.index > ref_date]
        constraints = ConstraintsBuilder(conf, data_loader.universe)
        market_data = MarketData(df_returns_train)
        hrp = HierarchicalRiskParity(market_data, constraints, conf)
        hrp.optimize()
        hrp.evaluate()
        mv = MeanVariance(market_data, constraints, conf)
        mv.optimize()
        mv.evaluate()
        print('optimized {}/{}'.format(i+1, N_PORTFOLIOS))
        hrp_predicted.append(hrp.variance)
        mv_predicted.append(mv.variance)
        hrp_real.append(np.var((df_returns_test * hrp.weights).sum(axis=1)))
        mv_real.append(np.var((df_returns_test * mv.weights).sum(axis=1)))

    plt.rcParams.update({'figure.figsize': [12, 10], 'font.size': 14})

    # ones = np.linspace(0, max(max(hrp_real), max(mv_real)), 100)
    ones_mv = np.linspace(0, max(max(mv_predicted), max(mv_real)), 100)
    ones_hrp = np.linspace(0, max(max(hrp_real), max(hrp_predicted)), 100)
    ones_real = np.linspace(0, max(max(hrp_real), max(mv_real)), 100)
    ones_pred = np.linspace(0, max(max(hrp_predicted), max(mv_predicted)), 100)
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(nrows=2, ncols=2)
    # plt.style.use('seaborn-darkgrid')
    ax1.scatter(x=mv_predicted, y=mv_real, color='orange', alpha=.5)
    ax1.plot(ones_mv, ones_mv, color='blue')
    # ax1.set_xticks(list(np.arange(7, 31, 2)))
    ax1.set_xlabel('variance predicted')
    ax1.set_ylabel('variance real')
    ax1.set_title('Mean-Variance')
    ax2.scatter(x=hrp_predicted, y=hrp_real, color='red', alpha=.5)
    ax2.plot(ones_hrp, ones_hrp, color='blue')
    ax2.set_title('HRP')
    # ax2.set_xticks(list(np.arange(7, 31, 2)))
    ax2.set_xlabel('variance predicted')
    ax2.set_ylabel('variance real')
    ax3.scatter(hrp_predicted, mv_predicted, color='grey', alpha=.5)
    ax3.plot(ones_pred, ones_pred, color='blue')
    ax3.set_title('HRP vs. MV predicted')
    ax3.set_xlabel('HRP variance predicted')
    ax3.set_ylabel('MV variance predicted')
    ax4.scatter(hrp_real, mv_real, color='green', alpha=.5)
    ax4.plot(ones_real, ones_real, color='blue')
    ax4.set_title('HRP vs. MV real')
    ax4.set_xlabel('HRP variance real')
    ax4.set_ylabel('MV variance real')
    plt.tight_layout()
    plt.savefig('backtesting/mv_vs_hrp.png')

    # without outliers:
    hrp_max = np.percentile(hrp_real, 99)
    mv_max = np.percentile(mv_real, 99)
    hrp_real_no, mv_real_no, hrp_predicted_no, mv_predicted_no = [], [], [], []
    for i in range(len(hrp_real)):
        if (hrp_real[i] < hrp_max) and (mv_real[i] < mv_max):
            hrp_real_no.append(hrp_real[i])
            mv_real_no.append(mv_real[i])
            hrp_predicted_no.append(hrp_predicted[i])
            mv_predicted_no.append(mv_predicted[i])

    ones_no = np.linspace(0, max(max(hrp_real_no), max(mv_real_no)), 100)
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(nrows=2, ncols=2)
    ax1.scatter(x=mv_predicted_no, y=mv_real_no, color='orange', alpha=.5)
    ax1.plot(ones_no, ones_no, color='blue')
    ax1.set_xlabel('variance predicted')
    ax1.set_ylabel('variance real')
    ax1.set_title('Mean-Variance')
    ax2.scatter(x=hrp_predicted_no, y=hrp_real_no, color='red', alpha=.5)
    ax2.plot(ones_no, ones_no, color='blue')
    ax2.set_title('HRP')
    ax2.set_xlabel('variance predicted')
    ax2.set_ylabel('variance real')
    ax3.scatter(hrp_predicted_no, mv_predicted_no, color='grey', alpha=.5)
    ax3.plot(ones_no, ones_no, color='blue')
    ax3.set_title('HRP vs. MV predicted')
    ax3.set_xlabel('HRP variance predicted')
    ax3.set_ylabel('MV variance predicted')
    ax4.scatter(hrp_real_no, mv_real_no, color='green', alpha=.5)
    ax4.plot(ones_no, ones_no, color='blue')
    ax4.set_title('HRP vs. MV real')
    ax4.set_xlabel('HRP variance real')
    ax4.set_ylabel('MV variance real')
    plt.tight_layout()
    plt.savefig('backtesting/mv_vs_hrp_no_outliers.png')




if __name__ == '__main__':
    main()
