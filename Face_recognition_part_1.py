#!/usr/bin/env python
# coding: utf-8

# # https://www.kaggle.com/saidakbarp/face-recognition-part-1

# In[9]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#Visiulazation
import matplotlib.pyplot as plt
#image processing
import cv2
#extracting zippped file
import tarfile
#systems
import os
print(os.listdir("/home/hduser/jupyter"))


# In[10]:


#example
imgg="/home/hduser/jupyter/Face_Detection_with_OpenCV/b97ea33b5842c7894b804923c6c05580.jpg"
imgg2 = "/home/hduser/jupyter/Face_Comparison_deepface/1.jpg"
celeb=cv2.imread(imgg2)


# In[11]:


def show_image(image):
    plt.figure(figsize=(8,5))
    #Before showing image, bgr color order transformed to rgb order
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()    


# In[12]:


show_image(celeb)


# In[13]:


# Our face detection function that uses haarcascade from OpenCV
def face_detection(img):
    face_cascade = cv2.CascadeClassifier('/home/hduser/jupyter/face-recognition/haarcascade_frontalface_alt.xml')
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    print('Number of faces detected:', len(faces))
        
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #img = img[y:y+h, x:x+w] # for cropping
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv_rgb


# In[14]:


#imgg2=cv2.imread("/kaggle/input/photos/ben.jpg")
a=face_detection(celeb)
plt.imshow(a)
plt.show() 

# as shown below, the library is not detecting this particular face angle of Ben Afflek


# In[15]:


plt.figure(figsize=(15,18))
img=cv2.imread("/home/hduser/jupyter/Face_Detection_with_OpenCV/b97ea33b5842c7894b804923c6c05580.jpg")
c=face_detection(img)
plt.imshow(c)
plt.show()

