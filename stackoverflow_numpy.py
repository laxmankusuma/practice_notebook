#!/usr/bin/env python
# coding: utf-8

# # stackoverlow most voted questions regading numpy

# ### How to print the full NumPy array, without truncation?

# In[1]:


# 1
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)


# In[2]:


# 2
import numpy as np
np.set_printoptions(threshold=np.inf)


# ### Dump a NumPy array into a csv file

# In[3]:


import numpy
a = numpy.asarray([ [1,2,3], [4,5,6], [7,8,9] ])
numpy.savetxt("foo.csv", a, delimiter=",")


# ### How can the Euclidean distance be calculated with NumPy?

# In[4]:


from scipy.spatial import distance
a = (1, 2, 3)
b = (4, 5, 6)
dist = distance.euclidean(a, b)
dist


# ### How do I get indices of N maximum values in a NumPy array?

# In[5]:


import numpy as np

arr = np.array([1, 3, 2, 4, 5])
arr


# In[6]:


arr.argsort()[-3:][::-1]


# In[7]:


arr[arr.argsort()[-3:][::-1]]


# In[8]:


a = np.array([9, 4, 4, 3, 3, 9, 0, 4, 6, 0])
a


# In[9]:


ind = np.argpartition(a, -4)[-4:]
ind


# In[10]:


a[ind]


# ### Convert pandas dataframe to NumPy array

# In[11]:


import numpy as np
import pandas as pd

index = [1, 2, 3, 4, 5, 6, 7]
a = [np.nan, np.nan, np.nan, 0.1, 0.1, 0.1, 0.1]
b = [0.2, np.nan, 0.2, 0.2, 0.2, np.nan, np.nan]
c = [np.nan, 0.5, 0.5, np.nan, 0.5, 0.5, np.nan]
df = pd.DataFrame({'A': a, 'B': b, 'C': c}, index=index)
df


# In[12]:


df.values


# ### Is there a NumPy function to return the first index of something in an array?

# In[13]:


t = np.array([1, 1, 1, 2, 2, 3, 8, 3, 8, 8])
np.nonzero(t == 8)


# In[14]:


np.nonzero(t == 8)[0][0]


# ### What does -1 mean in numpy reshape?

# In[15]:


z = np.array([[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12]])
z.shape


# In[16]:


z.reshape(-1)


# In[17]:


z.reshape(-1).shape


# In[18]:


z.reshape(-1,1)


# In[19]:


z.reshape(-1,1).shape


# In[20]:


z.reshape(-1, 2)


# In[21]:


z.reshape(-1, 2).shape


# In[22]:


z.reshape(1,-1)


# In[23]:


z.reshape(1,-1).shape


# In[24]:


z.reshape(2, -1)


# In[25]:


z.reshape(2, -1).shape


# In[26]:


z.reshape(3, -1)


# In[27]:


z.reshape(3, -1).shape


# ### What are the advantages of NumPy over regular Python lists?

# NumPy is not just more efficient; it is also more convenient. You get a lot of vector and matrix operations for free, which sometimes allow one to avoid unnecessary work. And they are also efficiently implemented.
# 
# For example, you could read your cube directly from a file into an array:

# x = numpy.fromfile(file=open("data"), dtype=float).reshape((100, 100, 100))

# Sum along the second dimension:
# 
# s = x.sum(axis=1)

# Find which cells are above a threshold:
# 
# (x > 0.5).nonzero()

# Remove every even-indexed slice along the third dimension:
# 
# x[:, :, ::2]

# lso, many useful libraries work with NumPy arrays. For example, statistical analysis and visualization libraries.
# 
# Even if you don't have performance problems, learning NumPy is worth the effort.

# ### How to count the occurrence of certain item in an ndarray?

# In[28]:


a = numpy.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4])
a


# In[29]:


unique, counts = numpy.unique(a, return_counts=True)
print(unique, counts)


# In[30]:


dict(zip(unique, counts))


# In[31]:


import collections, numpy
a = numpy.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4])
collections.Counter(a)


# In[32]:


y = np.array([1, 2, 2, 2, 2, 0, 2, 3, 3, 3, 0, 0, 2, 2, 0])
np.count_nonzero(y == 1)


# In[33]:


np.count_nonzero(y == 2)


# In[34]:


np.count_nonzero(y == 3)


# ### Most efficient way to map function over numpy array

# In[35]:


import numpy as np
x = np.array([1, 2, 3, 4, 5])
squarer = lambda t: t ** 2
vfunc = np.vectorize(squarer)
vfunc(x)


# ### Numpy array dimensions

# In[36]:


import numpy as np
a = np.array([[1,2],[1,2]])
a


# In[37]:


a.shape


# In[38]:


a.ndim  # num of dimensions/axes, *Mathematics definition of dimension*


# In[39]:


np.shape(a)


# In[40]:


a.shape[0]


# In[41]:


a.shape[1]


# ### Sorting arrays in NumPy by column

# In[42]:


a = np.array([[1,2,3],[4,5,6],[0,0,1]])
a


# In[43]:


sorted(a, key=lambda a_entry: a_entry[1]) 


# ### Find nearest value in numpy array

# In[44]:


a = numpy.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4])
a


# In[45]:


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# In[46]:


print(find_nearest(a, 0.8))


# ### How to pretty-print a numpy.array without scientific notation and with given precision?

# In[47]:


import numpy as np
x=np.random.random(10)
print(x)


# In[48]:


np.set_printoptions(precision=3)
print(x)


# In[49]:


y=np.array([1.5e-10,1.5,1500])
print(y)


# In[50]:


np.set_printoptions(suppress=True)
print(y)


# In[51]:


x = np.random.random(10)
x


# In[52]:


with np.printoptions(precision=3, suppress=True):
    print(x)


# ### What is the purpose of meshgrid in Python / NumPy?

# In[53]:


xvalues = np.array([0, 1, 2, 3, 4]);
yvalues = np.array([0, 1, 2, 3, 4]);


# In[54]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt

xx, yy = np.meshgrid(xvalues, yvalues)
plt.plot(xx, yy, marker='.', color='k', linestyle='none')


# ### Difference between numpy.array shape (R, 1) and (R,)

# In[55]:


# 1 dimension with 2 elements, shape = (2,). 
# Note there's nothing after the comma.
z=np.array([  # start dimension
    10,       # not a dimension
    20        # not a dimension
])            # end dimension
print(z.shape)


# In[56]:


# 2 dimensions, each with 1 element, shape = (2,1)
w=np.array([  # start outer dimension 
    [10],     # element is in an inner dimension
    [20]      # element is in an inner dimension
])            # end outer dimension
print(w.shape)


# ### What does numpy.random.seed(0) do?

# In[57]:


# np.random.seed(0) makes the random numbers predictable
numpy.random.seed(0); numpy.random.rand(4)


# In[58]:


numpy.random.seed(0); numpy.random.rand(4)


# In[59]:


numpy.random.rand(4)


# In[60]:


numpy.random.rand(4)


# ### How do I create an empty array/matrix in NumPy?

# In[61]:


a = numpy.zeros(shape=(5,2))
a


# In[62]:


a[0] = [1,2]
a[1] = [2,3]
a


# ### How to convert 2D float numpy array to 2D int numpy array?

# In[63]:


x = np.array([[1.0, 2.3], [1.3, 2.9]])
x


# In[64]:


x.astype(int)


# In[65]:


x = np.array([[1.0,2.3],[1.3,2.9]])
x


# In[66]:


y = np.trunc(x)
y


# In[67]:


z = np.ceil(x)
z


# In[68]:


t = np.floor(x)
t


# In[69]:


a = np.rint(x)
a


# In[70]:


a.astype(int)


# In[71]:


y.astype(int)


# In[72]:


np.int_(y)


# ### How to add an extra column to a NumPy array

# In[73]:


import numpy as np
N = 10
a = np.random.rand(N,N)
print(a.shape)
b = np.zeros((N,N+1))
print(b.shape)
b[:,:-1] = a
print(b.shape)


# ### What is the difference between Numpy's array() and asarray() functions?

# In[74]:


A = numpy.matrix(numpy.ones((3,3)))
A


# In[75]:


numpy.array(A)[2]=2
A


# In[76]:


numpy.asarray(A)[2]=2
A


# ### Converting NumPy array into Python List structure?

# In[77]:


np.array([[1,2,3],[4,5,6]]).tolist()


# ### Converting between datetime, Timestamp and datetime64

# In[78]:


from datetime import datetime
import numpy as np
dt = datetime.utcnow()
dt


# In[79]:


dt64 = np.datetime64(dt)
dt64


# In[80]:


ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
ts


# In[81]:


datetime.utcfromtimestamp(ts)


# ### What is the difference between flatten and ravel functions in numpy?

# In[82]:


import numpy
a = numpy.array([[1,2],[3,4]])

r = numpy.ravel(a)
f = numpy.ndarray.flatten(a)  

print(id(a))
print(id(r))
print(id(f))

print(r)
print(f)

print("\nbase r:", r.base)
print("\nbase f:", f.base)


# ### What does axis in pandas mean?

# axis=0 means each row as a bulk
# 
# axis=1 means each column as a bulk

# ### How do I check which version of NumPy I'm using?

# In[83]:


numpy.version.version


# ### Concatenating two one-dimensional NumPy arrays

# In[84]:


a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
print(a)
print("\n")
print(b)
print("\n")
# Appending below last row
print(np.concatenate((a, b), axis=0))

print("\n")
# Appending after last column
print(np.concatenate((a, b.T), axis=1))    # Notice the transpose
print("\n")
# Flattening the final array
print(np.concatenate((a, b), axis=None))


# ### Most efficient way to reverse a numpy array

# In[85]:


a


# In[86]:


a[::-1]


# ### What is the difference between ndarray and array in numpy?

# numpy.array is a function that returns a numpy.ndarray. There is no object type numpy.array.

# ### Converting numpy dtypes to native python types

# In[87]:


import numpy as np

# for example, numpy.float32 -> python float
val = np.float32(0)
pyval = val.item()
print(type(pyval))         # <class 'float'>

# and similar...
type(np.float64(0).item()) # <class 'float'>
type(np.uint32(0).item())  # <class 'long'>
type(np.int16(0).item())   # <class 'int'>
type(np.cfloat(0).item())  # <class 'complex'>
type(np.datetime64(0, 'D').item())  # <class 'datetime.date'>
type(np.datetime64('2001-01-01 00:00:00').item())  # <class 'datetime.datetime'>
type(np.timedelta64(0, 'D').item()) # <class 'datetime.timedelta'>


# ### Better way to shuffle two numpy arrays in unison

# In[88]:


a = numpy.array([[[  0.,   1.,   2.],
                  [  3.,   4.,   5.]],

                 [[  6.,   7.,   8.],
                  [  9.,  10.,  11.]],

                 [[ 12.,  13.,  14.],
                  [ 15.,  16.,  17.]]])

b = numpy.array([[ 0.,  1.],
                 [ 2.,  3.],
                 [ 4.,  5.]])


# In[89]:


#We can now construct a single array containing all the data:
c = numpy.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]
c


# In[90]:


a2 = c[:, :a.size//len(a)].reshape(a.shape)
a2


# In[91]:


b2 = c[:, a.size//len(a):].reshape(b.shape)
b2


# ### NumPy array initialization (fill with identical values)

# In[92]:


np.full((3, 5), 7)


# ### How do I convert a numpy array to (and display) an image?

# In[93]:


from PIL import Image
import numpy as np

w, h = 512, 512
data = np.zeros((h, w, 3), dtype=np.uint8)
data[0:256, 0:256] = [255, 0, 0] # red patch in upper left
img = Image.fromarray(data, 'RGB')
# img.save('my.png')
img.show()


# In[94]:


from matplotlib import pyplot as plt
plt.imshow(data, interpolation='nearest')
plt.show()


# ### dropping infinite values from dataframes in pandas?

# In[95]:


df = pd.DataFrame([1, 2, np.inf, -np.inf])
df


# In[96]:


df.replace([np.inf, -np.inf], np.nan)


# In[ ]:




