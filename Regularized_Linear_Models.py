#!/usr/bin/env python
# coding: utf-8

# # Ref.
# # https://www.kaggle.com/apapiu/regularized-linear-models

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina' #set 'png' here when working on note book")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv("/home/hduser/jupyter/Comprehensive_data_exploration_with_Python/train.csv")
test = pd.read_csv("/home/hduser/jupyter/Comprehensive_data_exploration_with_Python/test.csv")


# In[3]:


train


# In[4]:


#concate train and test by eliminatin id and target columns
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                    test.loc[:,'MSSubClass':'SaleCondition']))


# In[5]:


all_data


# **Data preprocessing:**
# **---------------------------------------**
# 
# 
# First I'll transform the skewed numeric features by taking log(feature + 1) - this will make the features more normal
# 
# Create Dummy variables for the categorical features
# 
# Replace the numeric missing values (NaN's) with the mean of their respective columns

# In[6]:


matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train['SalePrice'], 'log(price + 1)':np.log1p(train['SalePrice'])})
prices.hist()


# In[7]:


#log transform the target:
train['SalePrice'] = np.log1p(train['SalePrice'])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
numeric_feats


# In[8]:


skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats


# In[9]:


skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats


# In[10]:


skewed_feats = skewed_feats.index
skewed_feats


# In[11]:


all_data[skewed_feats] = np.log1p(all_data[skewed_feats])


# In[12]:


all_data = pd.get_dummies(all_data)


# In[13]:


#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())


# In[14]:


#creating matrices for sklearn:
print(all_data.shape)
print(train.shape)

X_train = all_data[:train.shape[0]]
print(X_train.shape)
X_test = all_data[train.shape[0]:]
print(X_test.shape)
y = train['SalePrice']


# **Models**
# 
# Now we are going to use regularized linear regression models from the scikit learn module. I'm going to try both l_1(Lasso) and l_2(Ridge) regularization. I'll also define a function that returns the cross-validation rmse error so we can evaluate our models and pick the best tuning par

# In[15]:


from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring='neg_mean_squared_error', cv = 5))
    return(rmse)


# In[16]:


model_ridge = Ridge()


# **The main tuning parameter for the Ridge model is alpha - a regularization parameter that measures how flexible our model is. The higher the regularization the less prone our model will be to overfit. However it will also lose flexibility and might not capture all of the signal in the data.**

# In[17]:


alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean()
           for alpha in alphas]


# In[18]:


cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = 'Validation - Just Do It')
plt.xlabel('alpha')
plt.ylabel('rmse')


# Note the U-ish shaped curve above. When alpha is too large the regularization is too strong and the model cannot capture all the complexities in the data. If however we let the model be too flexible (alpha small) the model begins to overfit. **A value of alpha = 10 is about right based on the plot above.**

# In[19]:


cv_ridge.min()


# So for the Ridge regression we get a rmsle of about 0.00122

# **Let' try out the Lasso model.** We will do a slightly different approach here and use the built in Lasso CV to figure out the best alpha for us. For some reason the alphas in Lasso CV are really the inverse or the alphas in Ridge.

# In[20]:


model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)


# In[21]:


rmse_cv(model_lasso).mean()


# Another neat thing about the Lasso is that it does feature selection for you - setting coefficients of features it deems unimportant to zero. Let's take a look at the coefficients:

# In[22]:


coef = pd.Series(model_lasso.coef_, index=X_train.columns)


# In[23]:


print('lasso picked '+str(sum(coef!= 0)) +  " variables and eliminated the other " + str(sum(coef==0)) + " variables")


# One idea to try here is run Lasso a few times on boostrapped samples and see how stable the feature selection is.

# We can also take a look directly at what the most important coefficients are:

# In[24]:


imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
imp_coef


# In[25]:


matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")


# In[26]:


#let's look at the residuals as well:
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")


# The residual plot looks pretty good.To wrap it up let's predict on the test set
