#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import pandas as pd
from statistics import *
from scipy import stats


# In[2]:


DepA = [1184,1203,1219,1238,1243,1204,1269,1256,1156,1248]
DepB = [1136,1178,1212,1193,1226,1154,1230,1222,1161,1148]


# In[32]:


def t_test(DepA, DepB):
    sum_DepA = sum(DepA)
    sum_DepB = sum(DepB)
    
    mean_DepA = mean(DepA)
    mean_DepB = mean(DepB)
    
    std_dev_A = stdev(DepA)
    std_dev_B = stdev(DepB)
    
    var_DepA = np.var(DepA)
    var_DepB = np.var(DepB)
    
    n_1 = len(DepA)
    n_2 = len(DepB)
    
    deg_free = n_1 + n_2 - 2
    
    top = mean_DepA - mean_DepB
    
    bottom = var_DepA / n_1 + var_DepB / n_2
    
    bottoms = math.sqrt(bottom)
    
    t_value = top / bottoms
    
    proba = 0.05
    
    n = len(DepA)
    
    critical_T = stats.t.ppf(1 - proba, n)
    
  
    if t_value > critical_T:
        print("We reject the NULL hypothesis.")
    else:
        print("We accept the NULL hypothesis.")
    
    return t_value, critical_T


# In[33]:


t_test(DepA, DepB)


# In[ ]:





# In[ ]:




