import os
import pandas as pd

DATA_DIR = '/Users/aplle/Code/MathModeling/SC/电力负荷数据'
OUT_DIR = '/Users/aplle/Code/MathModeling/SC/preprocessed'

# sum_df = pd.DataFrame()
years = [2016, 2017, 2018]
spans = ['1hour', '30min', '5min']
regions = ['Commercial', 'Residential', 'Office', 'Public']

pass_existing = True

for yr in years:
    for root, dirs, files in os.walk(os.path.join(DATA_DIR, str(yr))):
        for name in files:
            if os.path.splitext(name)[1] != '.xlsx':
                # print(os.path.splitext(name))
                continue
            filepath = os.path.join(root, name)

            sp = [span for span in spans if span in name][0]
            rg = [region for region in regions if region in name][0]

            if pass_existing and os.path.exists(os.path.join(OUT_DIR, str(yr), sp, os.path.splitext(name)[0]+'.csv')):
                continue

            try:
                df = pd.read_excel(filepath)
            except:
                print(f"Error reading {filepath}")
                continue

            if len(df['Power (kW)'].dropna()) == 0:
                continue
            df['Year'] = yr
            dt = pd.to_datetime(name[:8], format="%Y%m%d")
            df['Month'] = dt.month
            df['Day'] = dt.day
            df['Weekday'] = dt.weekday()
            df['Span'], df['Region'] = sp, rg
            os.makedirs(os.path.join(OUT_DIR, str(yr), sp), exist_ok=True)
            df.to_csv(os.path.join(OUT_DIR, str(yr), sp, os.path.splitext(name)[0]+'.csv'), index=False)
            # sum_df = df.concat([sum_df, df], axis=0, join='outer')
