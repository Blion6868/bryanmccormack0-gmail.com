#!/usr/bin/env python
# coding: utf-8

# In[95]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping


# In[96]:


df = pd.read_csv(r"C:\Users\bryan\Desktop\heart.csv")


# In[97]:


df.head()


# In[98]:


X = df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach']].values
y = df['target'].values


# In[99]:


from sklearn.model_selection import train_test_split


# In[100]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# In[101]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[102]:


nn = tf.keras.Sequential()

nn.add(Dense(30, activation='relu'))

nn.add(Dropout(0.2))

nn.add(Dense(15, activation='relu'))

nn.add(Dropout(0.2))


nn.add(Dense(1, activation='sigmoid'))

nn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss',mode='min', verbose=1, patience=25)

nn.fit(X_train, y_train, epochs = 1000, validation_data=(X_test, y_test),
         callbacks=[early_stop])


# In[103]:


model_loss = pd.DataFrame(nn.history.history)
model_loss.plot()


# In[104]:


predictions = nn.predict_classes(X_test)


# In[105]:


from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


# In[ ]:




