import matplotlib.pyplot as plt
import datetime
from collections import defaultdict

from opt.optimizers.hierarchical_risk_parity import HierarchicalRiskParity
from opt.optimizers.mean_variance import MeanVariance
from opt.optimizers.one_over_n import OneOverN
from opt.optimizers.random_weights import RandomWeights

from opt.opt_utils.opt_utils import create_parser, get_symbols
from utils.data_loader import DataLoader
from opt.opt_utils.constraints_builder import ConstraintsBuilder
from utils.market_data import MarketData

CONF = {
  "optimizer": "HierarchicalRiskParity",
  "yrs_look_back": 4,
  "min_weight_constraints": {},
  "max_weight_constraints": {},
  "max_number_instruments": 6,
  "consider_returns": True,
  "risk_appetite": 1.0,
  "expected_returns": {
    "AAPL": 0.05,
    "GOOG": 0.06,
    "FB": 0.08,
    "AMZN": 0.1,
    "KO": 0.02,
    "BX": 0.03
  }
}
CONF_DATA = {
    "api": "local_sample_data",
    "function": "TIME_SERIES_DAILY",
    "path": "sample_data/",
    "symbols": "['AAPL', 'GOOG', 'FB', 'AMZN', 'KO', 'BX']",
    "apikey": "JHC56UFM63VB6RUJ",
    "outputsize": "full",
    "datatype": "csv"
}


def main():
    # get user inputs
    parser = create_parser()
    # args = parser.parse_args()
    # conf = get_config(args.conf)
    # data_conf = get_config(args.data_conf)
    conf = CONF
    data_conf = CONF_DATA

    # prepare optimization problem
    universe = get_symbols(data_conf['symbols'])
    constraints = ConstraintsBuilder(conf, universe)
    min_date = datetime.date.today() - datetime.timedelta(days=365 * conf['yrs_look_back'])
    data_loader = DataLoader(data_conf, universe, min_date)
    data_loader.load_data()
    market_data = MarketData(data_loader.df_returns)

    # run optimization:
    # 1 over n
    opt = RandomWeights(market_data, constraints, conf)
    random_results = defaultdict(list)
    for _ in range(5000):
        opt.optimize()
        opt.evaluate()
        random_results['risk'].append(opt.variance)
        random_results['return'].append(opt.exp_return)
    on = OneOverN(market_data, constraints, conf)
    on.optimize()
    on.evaluate()
    mv_ = defaultdict(list)
    hrp_ = defaultdict(list)
    for ra in [0, 0.5, 1, 2, 5, 10]:
        conf_ = conf.copy()
        conf_['risk_appetite'] = ra
        mv = MeanVariance(market_data, constraints, conf_)
        mv.optimize()
        mv.evaluate()
        mv_['risk'].append(mv.variance)
        mv_['return'].append(mv.exp_return)
        hrp = HierarchicalRiskParity(market_data, constraints, conf_)
        hrp.optimize()
        hrp.evaluate()
        hrp_['risk'].append(hrp.variance)
        hrp_['return'].append(hrp.exp_return)

    plt.scatter(random_results['risk'], random_results['return'], alpha=0.4, s=2, label='random')
    plt.scatter(on.variance, on.exp_return, color='black', label='1 over n')
    # plt.scatter(mv.variance, mv.exp_return, color='red', label='mean variance')
    # plt.scatter(hrp.variance, hrp.exp_return, color='orange', label='hrp')
    # plt.scatter(mv_['risk'], mv_['return'], color='red', label='mean variance')
    plt.plot(mv_['risk'], mv_['return'], color='orange', marker='h', label='mean variance')
    plt.plot(hrp_['risk'], hrp_['return'], color='red', marker='h', label='hrp')
    plt.xlabel('risk')
    plt.ylabel('return')
    plt.legend()
    plt.savefig('backtesting/fig.png')


if __name__ == '__main__':
    main()
