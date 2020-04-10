#imports
import pandas as pd

#read file
df = pd.read_csv(input("Copy path of file you want to audit: "))

#check for certain condition (Y)
df2 = df.loc[(df['Track Item']=='Y')]
print(len(df2))

#function that determines to sample the category or subcategory
def simple_sample(df2):
    if df['Subcategory'].isnull().all():
        col_name = 'Subcategory'
    else:
        col_name = 'Category'
#sub-function that determines whether to sample 1, 3, or 5%, depending on the length of the catalog   
    def sub_sample(df2, col_name, frac):
        return df.groupby(col_name).apply(lambda x: x.sample(n=2) if x.size*frac < 2 else x.sample(frac=frac))
    if df2.shape[0] >= 15000:
        return sub_sample(df, col_name, 0.05)
    elif df2.shape[0] >= 10000:
        return sub_sample(df, col_name, 0.03)
    if df2.shape[0] >= 1500:
        return sub_sample(df, col_name, 0.01)
    else:
        return None

#results of function simple_sample
final = simple_sample(df2)

#compares original dataframe to the varaible final and marks items thst need to be audited as "Audit"
df.loc[df['Retailer Item ID'].isin(final['Retailer Item ID']), 'Track Item'] =='Audit'

#converts results to a csv file
df.to_csv('Test.csv',index=False)
