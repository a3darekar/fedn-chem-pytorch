#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split


# In[2]:


data = pd.read_csv('out3.csv')


# In[4]:


ids = data['molecule_id'].unique()


# In[25]:


train_set, test_set = [], []
for molecule_id in ids:
	df = data[(data.molecule_id == molecule_id)]
	train, test = train_test_split(df, test_size=0.2)
	train_set.append(train.values.tolist())
	test_set.append(test.values.tolist())


# In[28]:


train = np.array(list(itertools.chain(*train_set)))
print("train shape:", train.shape)
test = np.array(list(itertools.chain(*test_set)))
print("test shape:", test.shape)


# In[40]:


x_train, y_train = train[:,:-1], train[:,-1:]
print(x_train.shape, y_train.shape)

x_test, y_test = test[:,:-1], test[:,-1:]
print(x_test.shape, y_test.shape)


# In[50]:


np.savez("nmrshift",x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

