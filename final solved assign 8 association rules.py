#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# # install mlxtend

# In[4]:


# pip install mlxtend


# In[5]:


from mlxtend.frequent_patterns import apriori,association_rules


# In[6]:


df=pd.read_csv('C:/Users/RIG1/Desktop/DS ASSIGNMENTS/QUESTIONS -all assignments/ASS 8/book.csv')
df.head()


# # counting different-different combinations of frequencies of item-set

# In[9]:


frequent_item=apriori(df,min_support=0.1,use_colnames=True)
frequent_item


# # NOW CREATE ASSOCIATION RULE USING ABOVE freq_items list

# In[12]:


rules=association_rules(frequent_item,metric='lift',min_threshold=0.7)# lift= lift ratio


# In[13]:


rules


# # sorting top 20 rules based on lift

# In[14]:


rules.sort_values('lift',ascending=False)[0:20]


# # Now, just checking a list of all the rules whose lift ratio is greater is than >1

# In[15]:


rules[rules.lift>1]

# lift----lift ration > 1 is good. it means dependency between 2 items is more.so, the rule is good.


# In[ ]:




