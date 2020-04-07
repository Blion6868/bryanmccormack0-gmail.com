import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_excel(r"C:\Users\bryan\Desktop\Computer Stuff\pima-data.xlsx")

feature_col_names = ['num_preg','glucose_conc','diastolic_bp','thickness','insulin','bmi','diab_pred','age']
predicted_class_names = ['diabetes']

X = df[feature_col_names].values
y = df[predicted_class_names].values 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state = 0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred  = classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

X_new2 = [[6,148,72,35,0,33.6,0.627,50]]

X_new3 = classifier.predict(X_new2)

X_new4 = [[2,92,68,30,95,29.4,0.178,33]]

X_new5 = classifier.predict(X_new4)


