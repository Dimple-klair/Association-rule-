#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# for scaling/normalizing/z-score data import scale as well
from sklearn.preprocessing import scale


# In[2]:


df=pd.read_csv('C:/Users/RIG1/Desktop/DS ASSIGNMENTS/QUESTIONS -all assignments/ASS 9/wine.csv')
df.head()


# In[3]:


df['Type'].value_counts()


# WE HAVE THREE DIFFERENT VALUES IN 'Type' COLUMN


# # for scaling Dataframe
# # first slice the dataframe
# 
# # take ------- from 1st col to end col
# # avoid ----- 0th column ('Type') 

# In[4]:


df1=df.iloc[:,1:]


# In[5]:


df1


# In[6]:


df1.shape[0]


# In[7]:


df1.shape[1]

# now there are 13 columns instead of 14


# In[8]:


df1.shape


# In[9]:


df1.info()


# In[10]:


df1.describe()


# # Converting data(df1) to numpy array

# In[11]:


df1=df.values
df1


# # Normalizing the  numerical data

# In[12]:


df_normal=scale(df1)
df_normal


# # now on this above normalized data/array -------- apply PCA

# In[13]:


pca=PCA()


# In[14]:


pca_values=pca.fit_transform(df_normal)
pca_values


# # PCA Components matrix or we can say -------- covariance Matrix

# In[15]:


#pca.components_

pca=PCA(n_components=13)
pca_values=pca.fit_transform(df_normal)


# In[16]:


# lets check the variance ratio now:-

var = pca.explained_variance_ratio_
var


# In[17]:


var1=np.cumsum(np.round(var,decimals=4)*100)
var1


# # now plotting pca components we obtained

# In[18]:


plt.plot(var1,color ='red')


# In[19]:


pca_values[:,0:2]


# # now plot the columns

# In[21]:


x=pca_values[:,0:1]
y=pca_values[:,1:2]


# In[22]:


plt.scatter(x,y);


# In[23]:


# now label the above dots realted to its type
# final df


# In[24]:


final_df=pd.concat([df['Type'],pd.DataFrame(pca_values[:,0:2],columns=['pc1','pc2'])],axis=1)


# In[25]:


final_df


# # final scatter plot

# In[26]:


import seaborn as sns


# In[27]:


sns.scatterplot(data=final_df,x='pc1',y='pc2',hue='Type')


# # Now trying with hierarchichal clustering 

# In[28]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# ## create dendrogram first on normalized data
# ## As we already have normalized data, then, create Dendrograms

# In[33]:


plt.figure(figsize=(8,8))
den=sch.dendrogram(sch.linkage(df_normal,method='complete'))


# In[34]:


clust=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='complete')


# In[35]:


clust


# ## save clusters for chart purpose

# In[36]:


y_pred=clust.fit_predict(df_normal)


# In[37]:


y_pred
# 4 clusters------------ 0,1,2,3


# In[41]:


clusters=pd.DataFrame(y_pred,columns=['cluster_col'])
clusters


# In[44]:


#counting no of values i our clusters
clusters['cluster_col'].value_counts()


# ## add clusters to our dataset

# In[45]:


df['clusterid']=clusters


# In[46]:


df


# # now trying with k-means cluster

# In[49]:


from sklearn.cluster import KMeans


# ### as data is already normalized in PCA 
# ### so build k-means cluster
# 
# ### but before this we have to find out  best-k-value using-- ELBOW PLOT

# In[51]:


wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,random_state=40)
    kmeans.fit(df_normal)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow curv')
plt.xlabel('number of clusters')
plt.ylabel('WCSS')
plt.show()


# # k=4
# # now using k=4, build k-means cluster algo

# In[54]:


new_cluster=KMeans(4,random_state=40)
new_cluster.fit(df_normal)


# In[55]:


new_cluster.labels_


# # assign clusters to the dataset

# In[57]:


df['new_clus_col']=new_cluster.labels_


# In[58]:


df


# In[59]:


df.groupby('clusterid').agg(['mean']).reset_index()


# we have 4 clusterids------ 0th cluster,1st,2nd,3rd


# In[60]:


df['clusterid'].value_counts()


# In[ ]:




