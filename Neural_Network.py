import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\bryan\Desktop\Machine_Learning(2)\Machine Learning A-Z New\Part 8 - Deep Learning\Section 39 - Artificial Neural Networks (ANN)\Churn_Modelling.csv")
X = df.iloc[:, 3:13].values
y = df.iloc[:, 13].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

#11 features and 11+1/2=6 nodes
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy = (1528+156)/2000

#Test 1
X_new = [[1, 1, 700, 0, 33, 2, 1367, 2, 0, 1, 80000]]
X_new2 = ct.fit_transform(X_new)
X_new3 = classifier.predict(X_new2)

#Test2
XX_new = [[1, 1, 700, 0, 33, 2, 1367, 2, 0, 1, 80000]]
XX_new2 = sc.transform(XX_new)
XX_new3 = classifier.predict(XX_new2)