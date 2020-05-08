#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[58]:


df = pd.read_csv(r"C:\Users\bryan\Desktop\heart.csv")


# In[59]:


df.head()


# In[73]:


'''age - age in years
sex - (1 = male; 0 = female)
cp - chest pain type
trestbps - resting blood pressure (in mm Hg on admission to the hospital)
chol - serum cholestoral in mg/dl
fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
restecg - resting electrocardiographic results
thalach - maximum heart rate achieved
exang - exercise induced angina (1 = yes; 0 = no)
oldpeak - ST depression induced by exercise relative to rest
slope - the slope of the peak exercise ST segment
ca - number of major vessels (0-3) colored by flourosopy
thal - 3 = normal; 6 = fixed defect; 7 = reversable defect
target - have disease or not (1=yes, 0=no)'''


# In[60]:


df.isnull().sum()


# In[61]:


sns.pairplot(df)


# In[62]:


df.describe()


# In[63]:


df.corr()


# In[64]:


sns.countplot(df['target'])


# In[71]:


sns.countplot(x="sex", data=df)


# In[72]:


ax = sns.countplot(x="target", hue="sex", data=df)


# In[5]:


fig = sns.FacetGrid(data=df, hue='sex',aspect=4)
fig.map(sns.kdeplot,'age',shade=True)
fig.add_legend()


# In[56]:


fig = sns.FacetGrid(data=df, hue='target',aspect=4)
fig.map(sns.kdeplot,'age',shade=True)
fig.add_legend()


# In[65]:


df.head()


# In[66]:


X = df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach']].values
y = df['target'].values


# In[67]:


X.shape


# In[68]:


y.shape


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


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


# In[21]:


X_new = [[60,1,0,125,258,0,0,141]]


# In[22]:


clf.predict(X_new)


# In[23]:


X_new2 = [[50,0,0,110,254,0,0,159]]


# In[24]:


clf.predict(X_new2)


# Linear Support Vector Machine

# In[47]:


from sklearn import svm

classifier = svm.SVC(kernel='linear', gamma='auto',probability=True)
classifier.fit(X_train, y_train)


# In[48]:


ypred = classifier.predict(X_test)


# In[49]:


from sklearn.metrics import classification_report

print(classification_report(y_test,ypred))

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, ypred)


# In[50]:


from sklearn.metrics import r2_score

print(r2_score(ypred, y_test))


# In[51]:


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

# In[37]:


from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
classifier2.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(classifier2, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[38]:


ypredd = classifier2.predict(X_test)


# In[39]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, ypredd)


# In[40]:


cm


# In[41]:


from sklearn.metrics import classification_report

print(classification_report(y_test,ypredd))


# In[42]:


X_new = [[60,1,0,125,258,0,0,141]]
classifier2.predict(X_new)


# In[43]:


X_new2 = [[50,0,0,110,254,0,0,159]]
classifier2.predict(X_new2)


# In[44]:


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




