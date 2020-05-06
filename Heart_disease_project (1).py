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


df.head()


# In[4]:


df.tail()


# In[6]:


fig = sns.FacetGrid(data=df, hue='sex',aspect=4)
fig.map(sns.kdeplot,'age',shade=True)

#fig.set(xlim=(20,oldest))

fig.add_legend()


# In[7]:


df.isnull().sum()


# In[8]:


sns.countplot(df['target'])


# In[26]:


sns.pairplot(df)


# In[10]:


df.describe()


# In[11]:


df.corr()


# In[12]:


df.head()


# In[13]:


X = df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach']].values
y = df['target'].values


# In[14]:


X.shape


# In[15]:


y.shape


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# In[31]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_test, y_test)


# In[32]:


y_pred = clf.predict(X_test)
y_pred


# In[33]:


from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)


# In[34]:


from sklearn.metrics import r2_score

print(r2_score(y_pred, y_test))


# In[42]:


X_new = [[60,1,0,125,258,0,0,141]]


# In[43]:


clf.predict(X_new)


# In[44]:


X_new2 = [[50,0,0,110,254,0,0,159]]


# In[45]:


clf.predict(X_new2)


# SVM model for heart disease dataset

# In[57]:


from sklearn import svm
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)


# In[58]:


ypred = classifier.predict(X_test)


# In[59]:


from sklearn.metrics import classification_report

print(classification_report(y_test,ypred))

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, ypred)


# In[60]:


from sklearn.metrics import r2_score

print(r2_score(ypred, y_test))


# Random Forest

# In[77]:


from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier2.fit(X_train, y_train)


# In[78]:


ypredd = classifier2.predict(X_test)


# In[79]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, ypredd)


# In[80]:


cm


# In[81]:


from sklearn.metrics import classification_report

print(classification_report(y_test,ypredd))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




