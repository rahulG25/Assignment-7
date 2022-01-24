#!/usr/bin/env python
# coding: utf-8

# In[19]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize


# In[20]:


# Import Dataset
airline=pd.read_csv('EastWestAirlines.csv')
airline


# In[21]:


airline.info()


# In[22]:


airline2=airline.drop(['ID#'],axis=1)
airline2


# In[23]:


# Normalize heterogenous numerical data 
airline2_norm=pd.DataFrame(normalize(airline2),columns=airline2.columns)
airline2_norm


# In[24]:


# Create Dendrograms
plt.figure(figsize=(10, 7))  
dendograms=sch.dendrogram(sch.linkage(airline2_norm,'complete'))


# In[25]:


# Create Clusters (y)
hclusters=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
hclusters


# In[26]:


y=pd.DataFrame(hclusters.fit_predict(airline2_norm),columns=['clustersid'])
y['clustersid'].value_counts()


# In[27]:


# Adding clusters to dataset
airline2['clustersid']=hclusters.labels_
airline2


# In[28]:


airline2.groupby('clustersid').agg(['mean']).reset_index()


# In[29]:


# Plot Clusters
plt.figure(figsize=(10, 7))  
plt.scatter(airline2['clustersid'],airline2['Balance'], c=hclusters.labels_) 

