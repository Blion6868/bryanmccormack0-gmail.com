#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv(r"C:\Users\bryan\Desktop\heart.csv")


# In[3]:


X = df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach']].values
y = df['target'].values


# In[8]:


from sklearn.model_selection import train_test_split


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[23]:


import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()


classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 8))

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, nb_epoch = 150)


# In[24]:


test_loss, test_acc = classifier.evaluate(X_test, y_test, verbose=2)

print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)


# In[29]:


y_predd = classifier.predict(X_test)
y_predd


# In[31]:


y_test


# In[47]:


X_new = [[62,0,0,138,294,1,1,106]]
X_new = np.array(X_new)


# In[48]:


classifier.predict(X_new)


# In[50]:


X_new2 = [[59,1,1,140,221,0,1,164]]
X_new2 = np.array(X_new2)
classifier.predict(X_new2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




