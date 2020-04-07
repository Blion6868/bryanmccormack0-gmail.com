
import pandas as pd

df = pd.read_csv(r"C:/Users/bryan/Desktop/Computer Stuff/ADDITIONAL_ASINS_P&G_6.13.19.csv")

df2 = df.loc[(df['Track Item']=='Y')]

def sample_per(df2):
    if len(df2) >= 15000:
        return (df2.groupby('Category').apply(lambda x: x.sample(frac=0.01)))
    elif len(df2) < 15000 and len(df2) > 10000:
        return (df2.groupby('Category').apply(lambda x: x.sample(frac=0.03)))
    else:
        return (df2.groupby('Category').apply(lambda x: x.sample(frac=0.05)))

final = sample_per(df2)
df.loc[df['Retailer Item ID'].isin(final['Retailer Item ID']), 'Track Item'] = 'Audit'

df.to_csv('Test_3_31_20.csv', index=False)
