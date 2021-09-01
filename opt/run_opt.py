import numpy as np
import datetime
import time

from opt.optimizers.hierarchical_risk_parity import HierarchicalRiskParity
from utils.utils import get_symbols, get_config
from utils.parsers import opt_parser
from utils.data_loader import DataLoader
from opt.opt_utils.constraints_builder import ConstraintsBuilder
from utils.market_data import MarketData


def main():
    start_time = time.time()

    # get user inputs
    parser = opt_parser()
    args = parser.parse_args()
    conf = get_config(args.conf)
    data_conf = get_config(args.data_conf)

    # prepare optimization problem
    universe = get_symbols(data_conf['symbols'])
    constraints = ConstraintsBuilder(conf, universe)
    returns_to_date = datetime.date.today()
    returns_from_date = returns_to_date - datetime.timedelta(days=365 * conf['yrs_look_back'])
    data_loader = DataLoader(data_conf, returns_from_date, returns_to_date)
    data_loader.load_data()
    market_data = MarketData(data_loader.df_returns)

    # run optimization:
    if conf['max_number_instruments'] < len(market_data.universe):
        hrp = HierarchicalRiskParity(market_data, constraints, conf)
        hrp.optimize()
        reduced_universe = hrp.weights.sort_values(ascending=False).index.tolist()[:conf['max_number_instruments']]
        market_data.reduce_market_data(reduced_universe)
        constraints.ajdust_universe(reduced_universe)
    optimizer = HierarchicalRiskParity(market_data, constraints, conf)
    # opt = MeanVariance(market_data, constraints, conf)
    optimizer.optimize()
    optimizer.evaluate()
    print('\noptimized in {} seconds'.format(np.round(time.time() - start_time, 5)))
    optimizer.print_result()


if __name__ == '__main__':
    main()
