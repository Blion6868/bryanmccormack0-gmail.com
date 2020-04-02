#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split


# In[26]:


df = pd.read_csv("C:/Users/bryanmccormack/Downloads/covid19_italy_province.csv")


# In[27]:


df.head()


# In[28]:


df.tail()


# In[29]:


df.plot()


# In[82]:


train = df[['ProvinceCode']]
train


# In[83]:


test = df[['infected']]
test


# In[84]:


X_test, X_train, y_test, y_train = train_test_split(train, test, test_size=.33, random_state=42)


# In[85]:


train = xgb.DMatrix(X_train, label=y_train)
test = xgb.DMatrix(X_test, label=y_test)


# In[90]:


param = {
    'max_depth':4,
    'eta':0.3,
    'objective': 'multi:softmax',
    'num_class': 4}
epochs = 10


# In[91]:


model = xgb.train(param, train, epochs)


# In[92]:


testArray=np.array([[69]])
test_individual = xgb.DMatrix(testArray)


# In[93]:


model.predict(test_individual)


# In[ ]:





# In[ ]:




