#!/usr/bin/env python
# coding: utf-8

# # by Abhishek Thakur

# # Supervised vs unsupervised learning

# • **Supervised data:** always has one or multiple targets associated with it.
# <br>
# • **Unsupervised data:** does not have any target variable.

# If the target is categorical, the problem becomes a classification problem. And if the target is a real number, the problem is defined as a regression problem.
# <br>
# <br>
# • **Classification:** predicting a category, e.g. dog or cat.
# <br>
# • **Regression:** predicting a value, e.g. house prices.

# Clustering is one of the approaches of Unsupervised problems.
# <br>
# To make sense of unsupervised problems, we can also use
# numerous decomposition techniques such as **Principal Component Analysis
# (PCA)**, **t-distributed Stochastic Neighbour Embedding (t-SNE) etc.**

# https://www.kaggle.com/arthurtok/interactive-intro-to-dimensionality-reduction

# In[1]:


import matplotlib.pyplot as plt # for plotting
import numpy as np # to handle the numerical arrays
import pandas as pd # to create dataframes from the numerical arrays
import seaborn as sns # for plotting

from sklearn import datasets # to get the data
from sklearn import manifold # to perform t-SNE

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = datasets.fetch_openml(
    'mnist_784',
    version=1,
    return_X_y=True
)

pixel_values, targets = data


# In[3]:


pixel_values


# 784 means 28*28 pixels(each records is one image)

# In[4]:


pixel_values.info()


# In[5]:


targets


# In[6]:


targets = targets.astype(int)


# In[7]:


targets


# We can visualize the samples in this dataset by reshaping them to their original
# shape and then plotting them using matplotlib.

# In[8]:


single_image = pixel_values.iloc[1, :].values.reshape(28, 28)

plt.imshow(single_image, cmap='gray')


# In[9]:


tsne = manifold.TSNE(n_components=2, random_state=42)

transformed_data = tsne.fit_transform(pixel_values.iloc[:3000, :])


# In[10]:


transformed_data


# In[11]:


len(transformed_data)


# the above step creates the t-SNE transformation of the data.
# <br>
# We use only two components as we can visualize them well in a two-dimensional setting.
# <br>
# The transformed_data, in this case, is an array of shape 3000x2 (3000 rows and 2 columns). A data like this can be converted to a pandas dataframe by calling pd.DataFrame on the array.

# In[12]:


tsne_df = pd.DataFrame(
    np.column_stack((transformed_data, targets[:3000])),
    columns=['x','y','targets']
)


# In[13]:


tsne_df #x and y are the two components from t-SNE decomposition and targets is the actual number


# In[14]:


grid = sns.FacetGrid(tsne_df, hue='targets', size=8)
grid.map(plt.scatter, 'x', 'y').add_legend()


# This above is one way of visualizing unsupervised datasets.
# <br>
# We can also do **k-means clustering** on the same dataset and see how it performs in an unsupervised setting. You have to find the number clusters by **cross-validation.**
# <br>
# MNIST is a supervised classification problem, and we converted it to an unsupervised problem only to check if it gives any kind of good results.
# <br>
# we do get good results with decomposition with t-SNE, the results would be even better if we use classification algorithms
