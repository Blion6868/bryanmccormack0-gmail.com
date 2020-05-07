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


# In[5]:


fig = sns.FacetGrid(data=df, hue='sex',aspect=4)
fig.map(sns.kdeplot,'age',shade=True)
fig.add_legend()


# In[6]:


df.isnull().sum()


# In[7]:


sns.countplot(df['target'])


# In[26]:


sns.pairplot(df)


# In[8]:


df.describe()


# In[9]:


df.corr()


# In[10]:


df.head()


# In[11]:


X = df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach']].values
y = df['target'].values


# In[12]:


X.shape


# In[13]:


y.shape


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# Logisitc Regression

# In[16]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_test, y_test)


# In[17]:


y_pred = clf.predict(X_test)
y_pred


# In[18]:


from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)


# In[19]:


from sklearn.metrics import r2_score

print(r2_score(y_pred, y_test))


# In[20]:


X_new = [[60,1,0,125,258,0,0,141]]


# In[21]:


clf.predict(X_new)


# In[22]:


X_new2 = [[50,0,0,110,254,0,0,159]]


# In[23]:


clf.predict(X_new2)


# Linear Support Vector Machine

# In[69]:


from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0, probability=True)
classifier.fit(X_train, y_train)


# In[70]:


ypred = classifier.predict(X_test)


# In[71]:


from sklearn.metrics import classification_report

print(classification_report(y_test,ypred))

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, ypred)


# In[60]:


from sklearn.metrics import r2_score

print(r2_score(ypred, y_test))


# Random Forest

# In[28]:


from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier2.fit(X_train, y_train)


# In[29]:


ypredd = classifier2.predict(X_test)


# In[30]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, ypredd)


# In[31]:


cm


# In[32]:


from sklearn.metrics import classification_report

print(classification_report(y_test,ypredd))


# In[33]:


X_new = [[60,1,0,125,258,0,0,141]]
classifier2.predict(X_new)


# In[72]:


#ROC curve
ns_probs = [0 for _ in range(len(y_test))]

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

lr_probs = classifier.predict_proba(X_test)
lr_probs1 = lr_probs[:, 1]

ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs1)

print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
lr_auc = roc_auc_score(y_test, lr_probs1)

ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs1)

plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()


# In[73]:


#grid search to find best parameters
from sklearn.model_selection import GridSearchCV

param_grid = {'C':[0.1,1,10,100,1000], 'gamma':[1, 0.1, 0.01, 0.001, 000.1]}

grid = GridSearchCV(SVC(), param_grid, verbose=3)

grid.fit(X_train, y_train)
grid.best_params_
grid.best_estimator_
grid.best_score_

grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test, grid_predictions))
print('\n')
print(classification_report(y_test, grid_predictions))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




