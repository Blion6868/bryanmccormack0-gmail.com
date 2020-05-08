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


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# In[56]:


import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()


classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'sigmoid', input_dim = 8))

classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'sigmoid'))

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'sigmoid'))

classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'sigmoid'))

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, nb_epoch = 100)


# In[57]:


test_loss, test_acc = classifier.evaluate(X_test, y_test, verbose=2)

print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)


# In[58]:


y_predd = classifier.predict(X_test)
y_predd


# In[59]:


y_test


# In[60]:


X_new = [[62,0,0,138,294,1,1,106]]
X_new = np.array(X_new)


# In[61]:


classifier.predict(X_new)


# In[62]:


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




