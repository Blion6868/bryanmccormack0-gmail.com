import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:/Users/bryanmccormack/Downloads/covid19_italy_province.csv")

train = df[['ProvinceCode']]
test = df[['infected']]

X_test, X_train, y_test, y_train = train_test_split(train, test, test_size=.33, random_state=42)

train = xgb.DMatrix(X_train, label=y_train)
test = xgb.DMatrix(X_test, label=y_test)

param = {
    'max_depth':4,
    'eta':0.3,
    'objective': 'multi:softmax',
    'num_class': 4}
epochs = 10

model = xgb.train(param, train, epochs)

testArray=np.array([[69]])
test_individual = xgb.DMatrix(testArray)

model.predict(test_individual)
