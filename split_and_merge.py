import pandas as pd
import os
import datetime

DATA_DIR = '/Users/aplle/Code/MathModeling/SC/preprocessed'
OUT_DIR = '/Users/aplle/Code/MathModeling/SC/dataset1'

years = [2016, 2017, 2018]
spans = ['1hour', '30min', '5min']
regions = ['Commercial', 'Residential', 'Office', 'Public']

def iterate_date(start: datetime.date, end: datetime.date):
    tm = start
    while tm <= end:
        yield tm
        tm += datetime.timedelta(days=1)

train_df = pd.DataFrame()
eval_df = pd.DataFrame()
test_df = pd.DataFrame()

for dt in iterate_date(datetime.date(2016, 1, 1), datetime.date(2018, 12, 31)):
    yr = dt.year
    for region in regions:
        min5_path = os.path.join(DATA_DIR, str(yr), '5min', f"{dt.strftime('%Y%m%d')}_5min_{region}.csv")
        min30_path = os.path.join(DATA_DIR, str(yr), '30min', f"{dt.strftime('%Y%m%d')}_30min_{region}.csv")
        hour1_path = os.path.join(DATA_DIR, str(yr), '1hour', f"{dt.strftime('%Y%m%d')}_1hour_{region}.csv")
        if os.path.exists(min5_path):
            df = pd.read_csv(min5_path)
        elif os.path.exists(min30_path):
            df = pd.read_csv(min30_path)
        elif os.path.exists(hour1_path):
            df = pd.read_csv(hour1_path)
        else:
            continue

        if yr <= 2017 or dt.month <= 2:
            train_df = pd.concat([train_df, df], axis=0, join='outer')
        elif dt.month <= 8:
            eval_df = pd.concat([eval_df, df], axis=0, join='outer')
        else:
            test_df = pd.concat([test_df, df], axis=0, join='outer')

os.makedirs(OUT_DIR, exist_ok=True)
train_df.to_csv(os.path.join(OUT_DIR, 'train.csv'), index=False)
eval_df.to_csv(os.path.join(OUT_DIR, 'eval.csv'), index=False)
test_df.to_csv(os.path.join(OUT_DIR, 'test.csv'), index=False)