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


# Data Processing and Feature Extraction

# In[4]:



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


# In[5]:


df.isnull().sum()


# In[61]:


sns.pairplot(df)


# In[6]:


df.describe()


# In[7]:


df.corr()


# In[8]:


sns.countplot(df['target'])


# In[9]:


sns.countplot(x="sex", data=df)


# In[10]:


ax = sns.countplot(x="target", hue="sex", data=df)


# In[11]:


fig = sns.FacetGrid(data=df, hue='sex',aspect=4)
fig.map(sns.kdeplot,'age',shade=True)
fig.add_legend()


# In[12]:


fig = sns.FacetGrid(data=df, hue='target',aspect=4)
fig.map(sns.kdeplot,'age',shade=True)
fig.add_legend()


# In[13]:


df.head()


# In[14]:


X = df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach']].values
y = df['target'].values


# In[15]:


X.shape


# In[16]:


y.shape


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# Logisitc Regression

# In[19]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_test, y_test)


# In[20]:


y_pred = clf.predict(X_test)
y_pred


# In[21]:


from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)


# In[22]:


from sklearn.metrics import r2_score

print(r2_score(y_pred, y_test))


# In[23]:


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


# In[24]:


X_new = [[60,1,0,125,258,0,0,141]]


# In[25]:


clf.predict(X_new)


# In[26]:


X_new2 = [[50,0,0,110,254,0,0,159]]


# In[27]:


clf.predict(X_new2)


# Linear Support Vector Machine

# In[28]:


from sklearn import svm

classifier = svm.SVC(kernel='linear', gamma='auto',probability=True)
classifier.fit(X_train, y_train)


# In[29]:


ypred = classifier.predict(X_test)


# In[30]:


from sklearn.metrics import classification_report

print(classification_report(y_test,ypred))

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, ypred)


# In[31]:


from sklearn.metrics import r2_score

print(r2_score(ypred, y_test))


# In[32]:


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

# In[33]:


from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
classifier2.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(classifier2, X, y, cv=10)
scores.mean()


# In[34]:


ypredd = classifier2.predict(X_test)


# In[35]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, ypredd)


# In[36]:


cm


# In[37]:


from sklearn.metrics import classification_report

print(classification_report(y_test,ypredd))


# In[38]:


X_new = [[60,1,0,125,258,0,0,141]]
classifier2.predict(X_new)


# In[39]:


X_new2 = [[50,0,0,110,254,0,0,159]]
classifier2.predict(X_new2)


# In[40]:


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


# In[44]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores)


# In[45]:


print(scores.mean())


# In[46]:


k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)


# In[49]:


plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')


# In[61]:


print("Logistic Regression: 80%" + "\n" + "SVM: 79%" + "\n" + "Random Forest: 99%" + "\n" + "KNeighbors: 96%")


# In[ ]:





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




