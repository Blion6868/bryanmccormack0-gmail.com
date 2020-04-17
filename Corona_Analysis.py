#!/usr/bin/env python
# coding: utf-8

# In[2]:


#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#read csv file from Kaggle
df = pd.read_csv(r"C:\Users\bryan\Desktop\Corona.csv")


# In[4]:


#inspect first 5 rows
df.head()


# In[5]:


#general statistical overview of data
df.describe()


# In[6]:


df.replace(np.nan,0)
df.fillna(0)


# In[7]:


#turn symptom onset into dattime for easier analysis
df['symptom_onset'] = pd.to_datetime(df["symptom_onset"])
df['exposure_start'] = pd.to_datetime(df['exposure_start'])
df['exposure_end'] = pd.to_datetime(df['exposure_end'])


# In[8]:


symptoms = df['symptom'].value_counts().nlargest(10)
symptoms


# In[9]:


countries = df['country'].value_counts().nlargest(10)
countries


# In[64]:


plt.hist(df['country'])
plt.xticks(rotation='vertical');


# In[51]:


df['exposure_start'].value_counts()

df['exposure_start'].hist()
plt.xticks(rotation='vertical')


# In[52]:


df['exposure_end'].value_counts()

df['exposure_end'].hist()
plt.xticks(rotation='vertical')


# In[ ]:


deaths = df.groupby('death').sum()
death


# In[12]:


#correlation between all columns
df.corr()


# In[13]:


#general statistical overview of data
df.describe()


# In[31]:


#create new column for the day of virus onset
df['Day'] = df['symptom_onset'].dt.day
df['Month'] = df['symptom_onset'].dt.month


# In[32]:


df.head()


# In[17]:


#histogram of symptom onset
sns.catplot('Day', data=df, kind='count', size=12)


# In[18]:


#bar chart for gender count for virus
sns.catplot('gender', data=df, kind='count', size=10)


# In[49]:


#plot for age by gender
fig = sns.FacetGrid(df,hue='gender',aspect=4, size=3)

fig.map(sns.kdeplot,'age',shade=True)

oldest = df['age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend();


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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




