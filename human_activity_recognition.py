#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from IPython.display import display
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(123)


# In[ ]:


from google.colab import files
uploaded = files.upload()


# In[ ]:


import io
Data = pd.read_csv(io.BytesIO(uploaded['train.csv']))
Data 


# In[ ]:


Data.sample(5)


# In[ ]:


Data.shape


# In[ ]:


Labels = Data['activity']
Labels


# In[ ]:


Data = Data.drop(['rn', 'activity'], axis = 1)


# In[ ]:


Labels_keys = Labels.unique().tolist()
Labels_keys


# In[ ]:


Labels = np.array(Labels)
Labels


# In[ ]:


Temp = pd.DataFrame(Data.isnull().sum())
display(Temp)


# In[ ]:


Temp.columns = ['Sum']


# In[ ]:


Temp


# In[ ]:


len((Temp.index[Temp['Sum'] > 0])) 


# In[ ]:


scaler = StandardScaler()
Data = scaler.fit_transform(Data)


# In[ ]:


ks = range(1, 20)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(Data)
    inertias.append(model.inertia_)


# In[ ]:


inertias


# In[ ]:


plt.figure(figsize=(8,5))
plt.style.use('bmh')
plt.plot(ks, inertias, '-o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(ks)
plt.show()


# In[ ]:


def k_means(n_clust, data_frame, true_labels):

  k_means = KMeans(n_clusters = n_clust, random_state=123, n_init=30)
  k_means.fit(data_frame)
  c_labels = k_means.labels_
  df = pd.DataFrame({'clust_label': c_labels, 'orig_label': true_labels.tolist()})
  ct = pd.crosstab(df['clust_label'], df['orig_label'])
  y_clust = k_means.predict(data_frame)
  display(ct)
  print('% 9s' % 'inertia  homo    compl   v-meas   ARI     AMI     silhouette')
  print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
      %(k_means.inertia_,
      homogeneity_score(true_labels, y_clust),
      completeness_score(true_labels, y_clust),
      v_measure_score(true_labels, y_clust),
      adjusted_rand_score(true_labels, y_clust),
      adjusted_mutual_info_score(true_labels, y_clust),
      silhouette_score(data_frame, y_clust, metric='euclidean')))


# In[ ]:


k_means(n_clust=2, data_frame=Data, true_labels=Labels)


# In[ ]:


k_means(n_clust=6, data_frame=Data, true_labels=Labels)


# In[ ]:


Labels_binary = Labels.copy()
for i in range(len(Labels_binary)):
    if (Labels_binary[i] == 'STANDING' or Labels_binary[i] == 'SITTING' or Labels_binary[i] == 'LAYING'):
        Labels_binary[i] = 0
    else:
        Labels_binary[i] = 1
Labels_binary = np.array(Labels_binary.astype(int))


# In[ ]:


Labels_binary


# In[ ]:


k_means(n_clust=2, data_frame=Data, true_labels=Labels_binary)


# In[ ]:


pca = PCA(random_state=123)
pca.fit(Data)
features = range(pca.n_components_)

plt.figure(figsize=(8,4))
plt.bar(features[:15], pca.explained_variance_[:15], color='lightskyblue')
plt.xlabel('PCA feature')
plt.ylabel('Variance')
plt.xticks(features[:15])
plt.show()


# In[ ]:


def pca_transform(n_comp):
    pca = PCA(n_components=n_comp, random_state=123)
    global Data_reduced
    Data_reduced = pca.fit_transform(Data)
    print('Shape of the new Data df: ' + str(Data_reduced.shape))


# In[ ]:


pca_transform(n_comp=1)
k_means(n_clust=2, data_frame=Data_reduced, true_labels=Labels_binary)


# In[ ]:


pca_transform(n_comp=2)
k_means(n_clust=2, data_frame=Data_reduced, true_labels=Labels_binary)


# In[ ]:




