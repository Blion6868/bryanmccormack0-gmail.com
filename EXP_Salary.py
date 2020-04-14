#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read file
df = pd.read_csv(r"C:\Users\*****\Desktop\salary.csv")

#X and y values
X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

#splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state = 0)

#model selection
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

#predciting on X_test
y_pred = reg.predict(X_test)

from sklearn.metrics import r2_score

print(r2_score(y_pred, y_test))

#new_predictions
X_new = [[5]]
print(reg.predict(X_new))

X_new1 = [[1]]
print(reg.predict(X_new1))

#plot
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, reg.predict(X_train), color='black')
plt.title('Salary / Experience--Training set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, color='blue')
plt.plot(X_train, reg.predict(X_train), color='green')
plt.title('Salary / Experience--Test set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
