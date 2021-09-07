import pandas as pd

INPUT_CSV = 'data_2021-09-04/NVDA_2021-09-06.csv'
OUTPUT_CSV = 'data_2021-09-04/NVDA_2021-09-06.csv'
SPLIT_DATE = '2021-07-20'
SPLIT_FACTOR = .25

data = pd.read_csv(INPUT_CSV)
data_before = data[data['timestamp'] < SPLIT_DATE]
data_before[['open', 'high', 'low', 'close']] *= SPLIT_FACTOR
data_after = data[data['timestamp'] >= SPLIT_DATE]
data_new = pd.concat([data_before, data_after])
data_new.to_csv(OUTPUT_CSV, index=False)
