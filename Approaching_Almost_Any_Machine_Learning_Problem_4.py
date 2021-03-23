#!/usr/bin/env python
# coding: utf-8

# # by Abhishek Thakur

# # Arranging machine learning projects

# In[1]:


# The inside of the project folder should look something like the following.
'''
├── input
│ ├── train.csv
│ └── test.csv
├── src
│ ├── create_folds.py
│ ├── train.py
│ ├── inference.py
│ ├── models.py
│ ├── config.py
│ └── model_dispatcher.py
├── models
│ ├── model_rf.bin
│ └── model_et.bin
├── notebooks
│ ├── exploration.ipynb
│ └── check_data.ipynb
├── README.md
└── LICENSE
'''


# input/: This folder consists of all the input files
# <br>
# src/: We will keep all the python scripts associated with the project here. If I talk
# about a python script, i.e. any *.py file, it is stored in the src folder.
# <br>
# models/: This folder keeps all the trained models.
# <br>
# notebooks/: All jupyter notebooks (i.e. any *.ipynb file) are stored in the notebooks
# folder.
# <br>
# README.md: This is a markdown file where you can describe your project and
# write instructions on how to train the model or to serve this in a production
# environment.
# <br>
# LICENSE: This is a simple text file that consists of a license for the project, such as
# MIT, Apache, etc.

# If the distribution of labels is quite good and even. We can thus use
# accuracy/F1 as metrics. This is the first step when approaching a machine learning
# problem: decide the metric!

# In[2]:


# src/train.py

import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree

def run(fold):
    # read the training data with folds
    df = pd.read_csv("/home/hduser/jupyter/winequality-red_n_folds.csv")
    # training data is where kfold is not equal to provided fold
    # also, note that we reset the index
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    
    # validation data is where kfold is equal to provided fold
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)
    
    # drop the label column from dataframe and convert it to
    # a numpy array by using .values.
    # target is label column in the dataframe
    x_train = df_train.drop('quality', axis=1).values
    y_train = df_train['quality'].values
    
    # similarly, for validation, we have
    x_valid = df_valid.drop("quality", axis=1).values
    y_valid = df_valid['quality'].values

    # initialize simple decision tree classifier from sklearn
    clf = tree.DecisionTreeClassifier()
    
    # fit the model on training data
    clf.fit(x_train, y_train)
    
    # create predictions for validation samples
    preds = clf.predict(x_valid)
    
    # calculate & print accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")
    
    # save the model
    joblib.dump(clf, f"/home/hduser/jupyter/dt_{fold}.bin")
    
    
if __name__ == "__main__":
    run(fold=0)
    run(fold=1)
    run(fold=2)
    run(fold=3)
    run(fold=4)


# You can run this script by calling **python train.py** in the console.

# below 5 files are creating in the location
# <br>
# dt_0.bin
# <br>
# dt_1.bin
# <br>
# dt_2.bin
# <br>
# dt_3.bin
# <br>
# dt_4.bin

# In[ ]:




