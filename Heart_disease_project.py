#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspe
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics


# In[2]:


df = pd.read_csv(r"C:\Users\bryan\Desktop\heart.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[84]:


fig = sns.FacetGrid(data=df, hue='sex',aspect=4)
fig.map(sns.kdeplot,'age',shade=True)

oldest = df['age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()


# In[5]:


df.isnull().sum()


# In[6]:


sns.countplot(df['target'])


# In[6]:


sns.pairplot(df)


# In[7]:


df.describe()


# In[8]:


df.corr()


# In[9]:


df.head()


# In[53]:


X = df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach']].values
y = df['target'].values


# In[54]:


X.shape


# In[55]:


y.shape


# In[56]:


from sklearn.model_selection import train_test_split


# In[57]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# In[58]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_test, y_test)


# In[59]:


y_pred = clf.predict(X_test)
y_pred


# In[73]:


from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)


# In[79]:


from sklearn.metrics import r2_score

print(r2_score(y_pred, y_test))


# In[74]:


X_new = [[52,1,0,125,212,0,1,168]]


# In[75]:


clf.predict(X_new)


# In[76]:


X_new2 = [[50,0,0,110,254,0,0,159]]


# In[77]:


clf.predict(X_new2)


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





# In[ ]:





# In[ ]:





# In[ ]:




