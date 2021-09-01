# Hierarchical Risk Parity - Portfolio optimization

The goal of this project is to assist you in creating an optimal portfolio (minimal variance & maximimum return) considering a personal universe, constraints 
and expectations. The portfolio is optimized using the Hieratchical Risk Parity algorithm (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678)

### High level idea: ###
1. find all the tickers of stocks of your interest through own research
2. use the "anlys" tool to analyze the recent performacne of your selected stocks -- this should help shortlisting your favorite stocks
3. specify all the constraints for your optimization problem, such as
 * universe of instruments / max number of instruments to consider
 * minimum / maximum weight of each instrument
 * risk appetite / expected returns
4. run optimization and receive weights for each instrument in yout universe

### Running instructions (unix terminal) ###
1. clone the repo
2. append the projects path to PYTHONPATH:
```
export PYTHONPATH="${PYTHONPATH}:/path/to/project/"
```
3. create a data-configuration json file (see the example: "configs/dataconf_load_data_alphavantage.json"). To load the input data, you can use the free 
Alphavantage API
   * specify the tickers of all stocks that you consider in your portfolio
   * specify your Alphavantage API-key (get it here: https://www.alphavantage.co/support/#api-key)

4. load the data:
```
python data/run_get_data.py --data_conf configs/<dataconf_load_data_alphavantage_example.json>   
```
5. create a data-configuration json file for the analysis (see the example "configs/dataconf_anly_local_example.json"). Use the downloaded CSV files from step (3)
```
python anly/run_anly.py --data_conf configs/<dataconf_anly_local_example.json>
```
6. specify the configurations for the optimization problem
   * create a data-configuration json file for the optimization problem (see the example "configs/dataconf_opt_local_example.json")
   * create an optimization configuration where you specify your constraints and expected returns (see the example "configs/optconf_hrp_example.json")
7. run the optimization problem:
```
python opt/run_opt.py --conf configs/optconf_hrp.json --data_conf configs/dataconf_opt_local_data.json
```
