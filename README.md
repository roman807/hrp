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

### Example running instructions (unix terminal) ###
1. clone the repo
2. create and activate the conda-environment and install the required libraries from the environment.yml file 
```
conda env create -f environment.yml
conda activate hrp-env
```
3. append the project path to PYTHONPATH:
```
export PYTHONPATH="${PYTHONPATH}:</path/to/project/>"
```
4. download the market data CSV files on your machine
* create a data-configuration json file (see the example: "configs/dataconf_load_data_alphavantage.json"). To download the market data, you can use the free 
Alphavantage API
   * specify the tickers of all stocks that you are interested in
   * specify your Alphavantage API-key (get it here: https://www.alphavantage.co/support/#api-key)

* run the command to download the data (use the name of your json config file):
```
python data/run_get_data.py --data_conf configs/dataconf_load_data_alphavantage_example.json 
```
5. run the analysys
* create a data-configuration json file for the analysis (see the example "configs/dataconf_anly_local_example.json"). Use the tickers from data you downloaded CSV files from step (4.)
* run the command:
```
python anlys/run_anlys.py --data_conf configs/dataconf_anly_local_example.json
```
* view the resulting dashboard in your browser on: ```http://localhost:8010/```
6. optimize your portfolio 
* specify the configurations for the optimization problem
   * create a data-configuration json file for the optimization problem (see the example "configs/dataconf_opt_local_example.json")
   * create an optimization configuration where you specify your constraints and expected returns (see the example "configs/optconf_hrp_example.json")
* run the optimization problem:
```
python opt/run_opt.py --conf configs/optconf_hrp.json --data_conf configs/dataconf_opt_local_data.json
```
