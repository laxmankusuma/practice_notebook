#!/usr/bin/env python
# coding: utf-8

# # https://www.youtube.com/watch?v=f9aV2Jfd5fc

# In[1]:


# pip install deepface


# In[2]:


from deepface import DeepFace
import cv2


# In[3]:


image1 = cv2.imread('/home/hduser/jupyter/Face_Comparison_deepface/1.jpg')
image2 = cv2.imread('/home/hduser/jupyter/Face_Comparison_deepface/2.jpg')


# In[4]:


DeepFace.verify(image1,image2)['verified']


# In[ ]:




