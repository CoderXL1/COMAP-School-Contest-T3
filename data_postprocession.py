import pandas as pd
import os

DATA_DIR = '/Users/aplle/Code/MathModeling/SC/dataset1'

train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
eval_df = pd.read_csv(f"{DATA_DIR}/eval.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

dfs = [train_df, eval_df, test_df]

for df in dfs:
    df['Time'] = pd.to_datetime(df['Time'])
    df['Days_from_NYD'] = df['Time'] - pd.to_datetime(df['Year'], format='%Y')
    df['Days_from_NYD'] = df['Days_from_NYD'].dt.days
    df['Time'] = df['Time'].dt.time

train_df.to_csv(os.path.join(DATA_DIR, 'train_processed.csv'), index=False)
eval_df.to_csv(os.path.join(DATA_DIR, 'eval_processed.csv'), index=False)
test_df.to_csv(os.path.join(DATA_DIR, 'test_processed.csv'), index=False)