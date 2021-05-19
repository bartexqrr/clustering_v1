#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, ward_tree, SpectralClustering

# ustawianie statyczne ziarna losowania ->
# zapewnienie powtarzalności generowanychn wyników
np.random.seed(1234)


# In[3]:


clean_df2 = pd.read_csv('clean_df2.csv')
clean_df2.shape


# In[4]:


input_df = np.array(clean_df2)[:, 1:]
input_df.shape


# In[5]:


# 2. podejście
# zbiór zredukowany o połowę

# od tego miejsca Spactral Clustering

reduced_df = clean_df2[::1]
reduced_df.shape


# In[ ]:


#te wyniki masz wygenerowane i zapisane, nie powtarzam tutaj obliczeń

models = [
    ('KM_2', KMeans(n_clusters=2)),
    ('KM_3', KMeans(n_clusters=3)),
    ('KM_4', KMeans(n_clusters=4)),
    ('KM_5', KMeans(n_clusters=5)),
    ('KM_6', KMeans(n_clusters=6)),
    ('AG_2', AgglomerativeClustering(n_clusters=2)),
    ('AG_3', AgglomerativeClustering(n_clusters=3)),
    ('AG_4', AgglomerativeClustering(n_clusters=4)),
    ('AG_5', AgglomerativeClustering(n_clusters=5)),
    ('AG_6', AgglomerativeClustering(n_clusters=6))
    
]

results_reduced = {}

for name, model in models:
    clf = model
    results_reduced[name] = clf.fit_predict(reduced_df)
    print(name, ' results:')
    print(pd.Series(results_reduced[name]).value_counts())
    print('\n')
    
pd.DataFrame(results_reduced).to_csv('results_reduced.csv')
results_reduced_df = pd.DataFrame(results_reduced)


# In[ ]:


models = [
    ('SC_2', SpectralClustering(n_clusters=2)),
    ('SC_3', SpectralClustering(n_clusters=3)),
    ('SC_4', SpectralClustering(n_clusters=4)),
    ('SC_5', SpectralClustering(n_clusters=5)),
    ('SC_6', SpectralClustering(n_clusters=6))
]

results_reduced_2 = {}

for name, model in models:
    clf = model
    results_reduced_2[name] = clf.fit_predict(reduced_df)
    print(name, ' results:')
    print(pd.Series(results_reduced[name]).value_counts())
    print('\n')
    
pd.DataFrame(results_reduced_2).to_csv('results_reduced_2.csv')
results_reduced_2_df = pd.DataFrame(results_reduced_2)


# In[ ]:





# In[ ]:






# In[ ]:





