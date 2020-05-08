#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv(r"C:\Users\bryan\Desktop\heart.csv")


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


fig = sns.FacetGrid(data=df, hue='sex',aspect=4)
fig.map(sns.kdeplot,'age',shade=True)
fig.add_legend()


# In[7]:


df.isnull().sum()


# In[8]:


sns.countplot(df['target'])


# In[26]:


sns.pairplot(df)


# In[9]:


df.describe()


# In[10]:


df.corr()


# In[11]:


df.head()


# In[12]:


X = df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach']].values
y = df['target'].values


# In[13]:


X.shape


# In[14]:


y.shape


# In[15]:


from sklearn.model_selection import train_test_split


# In[215]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# Logisitc Regression

# In[216]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_test, y_test)


# In[217]:


y_pred = clf.predict(X_test)
y_pred


# In[218]:


from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)


# In[219]:


from sklearn.metrics import r2_score

print(r2_score(y_pred, y_test))


# In[220]:


#ROC curve
ns_probs = [0 for _ in range(len(y_test))]

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

lr_probs = clf.predict_proba(X_test)
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


# In[221]:


X_new = [[60,1,0,125,258,0,0,141]]


# In[222]:


clf.predict(X_new)


# In[223]:


X_new2 = [[50,0,0,110,254,0,0,159]]


# In[224]:


clf.predict(X_new2)


# Linear Support Vector Machine

# In[225]:


from sklearn import svm

classifier = svm.SVC(kernel='linear', probability=True)
classifier.fit(X_train, y_train)


# In[226]:


ypred = classifier.predict(X_test)


# In[227]:


from sklearn.metrics import classification_report

print(classification_report(y_test,ypred))

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, ypred)


# In[228]:


from sklearn.metrics import r2_score

print(r2_score(ypred, y_test))


# In[229]:


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


# Random Forest

# In[234]:


from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
classifier2.fit(X_train, y_train)


# In[235]:


ypredd = classifier2.predict(X_test)


# In[236]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, ypredd)


# In[237]:


cm


# In[238]:


from sklearn.metrics import classification_report

print(classification_report(y_test,ypredd))


# In[239]:


X_new = [[60,1,0,125,258,0,0,141]]
classifier2.predict(X_new)


# In[240]:


#ROC curve
ns_probs = [0 for _ in range(len(y_test))]

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

lr_probs = classifier2.predict_proba(X_test)
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


# In[ ]:





# In[163]:





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




