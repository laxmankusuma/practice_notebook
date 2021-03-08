#!/usr/bin/env python
# coding: utf-8

# #REFERENCE
# 
# https://github.com/ageron/handson-ml2/blob/master/tools_numpy.ipynb

# In[1]:


import numpy as np


# In[2]:


np.zeros(5)


# In[3]:


np.zeros((3,4))


# In[4]:


a = np.zeros((3,4))
a


# In[5]:


a.shape


# In[6]:


a.shape[0]


# In[7]:


a.ndim  # equal to len(a.shape)


# In[8]:


a.size


# In[9]:


np.zeros((2,3,4))


# In[10]:


type(np.zeros((3,4)))


# In[11]:


np.ones((3,4))


# In[12]:


np.full((3,4), np.pi)


# In[13]:


np.empty((2,3))


# In[14]:


np.array([[1,2,3,4], [10, 20, 30, 40]])


# In[15]:


np.arange(1, 5)


# In[16]:


np.arange(1.0, 5.0)


# In[17]:


np.arange(1, 5, 0.5)


# In[18]:


print(np.arange(0, 5/3, 1/3)) # depending on floating point errors, the max value is 4/3 or 5/3.
print(np.arange(0, 5/3, 0.333333333))
print(np.arange(0, 5/3, 0.333333334))


# In[19]:


print(np.linspace(0, 5/3, 6))


# In[20]:


np.random.rand(3,4)


# In[21]:


np.random.randn(3,4)


# In[22]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[23]:


plt.hist(np.random.rand(100000), density=True, bins=100, histtype="step", color="blue", label="rand")
plt.hist(np.random.randn(100000), density=True, bins=100, histtype="step", color="red", label="randn")
plt.axis([-2.5, 2.5, 0, 1.1])
plt.legend(loc = "upper left")
plt.title("Random distributions")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()


# In[24]:


def my_function(z, y, x):
    return x * y + z

np.fromfunction(my_function, (3, 2, 10))


# In[25]:


c = np.arange(1, 5)
print(c.dtype, c)


# In[26]:


c = np.arange(1.0, 5.0)
print(c.dtype, c)


# In[27]:


d = np.arange(1, 5, dtype=np.complex64)
print(d.dtype, d)


# In[28]:


e = np.arange(1, 5, dtype=np.complex64)
e.itemsize


# In[29]:


f = np.array([[1,2],[1000, 2000]], dtype=np.int32)
f.data


# In[30]:


if (hasattr(f.data, "tobytes")):
    data_bytes = f.data.tobytes() # python 3
else:
    data_bytes = memoryview(f.data).tobytes() # python 2

data_bytes


# In[31]:


g = np.arange(24)
print(g)
print("Rank:", g.ndim)


# In[32]:


g.shape = (6, 4)
print(g)
print("Rank:", g.ndim)


# In[33]:


g.shape = (2, 3, 4)
print(g)
print("Rank:", g.ndim)


# In[34]:


g2 = g.reshape(4,6)
print(g2)
print("Rank:", g2.ndim)


# In[35]:


g2[1, 2] = 999 # [row,column]
g2


# In[36]:


g


# In[37]:


g.ravel()


# In[38]:


a = np.array([14, 23, 32, 41])
b = np.array([5,  4,  3,  2])
print("a + b  =", a + b)
print("a - b  =", a - b)
print("a * b  =", a * b)
print("a / b  =", a / b)
print("a // b  =", a // b)
print("a % b  =", a % b)
print("a ** b =", a ** b)


# Broadcasting
# 
# In general, when NumPy expects arrays of the same shape but finds that this is not the case, it applies the so-called broadcasting rules:

# In[39]:


h = np.arange(5).reshape(1, 1, 5)
h


# In[40]:


h + [10, 20, 30, 40, 50]  # same as: h + [[[10, 20, 30, 40, 50]]]


# In[41]:


k = np.arange(6).reshape(2, 3)
k


# In[42]:


k + [[100], [200]]  # same as: k + [[100, 100, 100], [200, 200, 200]]


# In[43]:


k + [100, 200, 300]  # after rule 1: [[100, 200, 300]], and after rule 2: [[100, 200, 300], [100, 200, 300]]


# In[44]:


k + 1000  # same as: k + [[1000, 1000, 1000], [1000, 1000, 1000]]


# In[45]:


try:
    k + [33, 44]
except ValueError as e:
    print(e)


# In[46]:


k1 = np.arange(0, 5, dtype=np.uint8)
print(k1.dtype, k1)


# In[47]:


k2 = k1 + np.array([5, 6, 7, 8, 9], dtype=np.int8)
print(k2.dtype, k2)


# In[48]:


k3 = k1 + 1.5
print(k3.dtype, k3)


# In[49]:


m = np.array([20, -5, 30, 40])
m < [15, 16, 35, 36]


# In[50]:


m < 25  # equivalent to m < [25, 25, 25, 25]


# In[51]:


m[m < 25]


# In[52]:


a = np.array([[-2.5, 3.1, 7], [10, 11, 12]])
print(a)
print("mean =", a.mean())


# In[53]:


for func in (a.min, a.max, a.sum, a.prod, a.std, a.var):
    print(func.__name__, "=", func())


# In[54]:


c=np.arange(24).reshape(2,3,4)
c


# In[55]:


c.sum(axis=0)  # sum across matrices


# In[56]:


c.sum(axis=1)  # sum across rows
#[0+4+8,1+5+9,2+6+10,3+7+11],[12+16+28,14+22+30,16+24+32,18+26+34]


# In[57]:


[0+4+8,1+5+9,2+6+10,3+7+11],[12+16+20,13+17+21,14+18+22,15+19+23]


# In[58]:


c.sum(axis=(0,2))  # sum across matrices and columns


# In[59]:


0+1+2+3 + 12+13+14+15, 4+5+6+7 + 16+17+18+19, 8+9+10+11 + 20+21+22+23


# In[60]:


a = np.array([[-2.5, 3.1, 7], [10, 11, 12]])
np.square(a)


# In[61]:


print("Original ndarray")
print(a)
for func in (np.abs, np.sqrt, np.exp, np.log, np.sign, np.ceil, np.modf, np.isnan, np.cos):
    print("\n", func.__name__)
    print(func(a))


# In[62]:


a = np.array([1, -2, 3, 4])
b = np.array([2, 8, -1, 7])
np.add(a, b)  # equivalent to a + b


# In[63]:


np.greater(a, b)  # equivalent to a > b


# In[64]:


np.maximum(a, b)


# In[65]:


np.copysign(a, b)


# In[66]:


a = np.array([1, 5, 3, 19, 13, 7, 3])
a[3]


# In[67]:


a[2:5]


# In[68]:


a[2:-1]


# In[69]:


a[:2]


# In[70]:


a[2::1]#skip 1-1=0 from indesx 2


# In[71]:


a[2::2]#skip 2-1=1 from indesx 2


# In[72]:


a[2::3]#skip 3-1=2 from indesx 2


# In[73]:


a[2::-1]#skip 1-1=0 from indesx 2(reverse order)


# In[74]:


a[4::-2]#skip 2-1=1 from indesx 4(reverse order)


# In[75]:


a


# In[76]:


a[3]=999
a


# In[77]:


a[2:5] = [997, 998, 999]
a


# In[78]:


a[2:5] = -1
a


# In[79]:


try:
    a[2:5] = [1,2,3,4,5,6]  # too long
except ValueError as e:
    print(e)


# In[80]:


try:
    del a[2:5]
except ValueError as e:
    print(e)


# In[81]:


a_slice = a[2:6]
a_slice[1] = 1000
a  # the original array was modified!


# In[82]:


a[3] = 2000
a_slice  # similarly, modifying the original array modifies the slice!


# In[83]:


another_slice = a[2:6].copy()
another_slice[1] = 3000
a  # the original array is untouched


# In[84]:


a[3] = 4000
another_slice  # similary, modifying the original array does not affect the slice copy


# In[85]:


a


# In[86]:


b = np.arange(48).reshape(4, 12)
b


# In[87]:


b[1, 2]  # row 1, col 2


# In[88]:


b[1, :]  # row 1, all columns


# In[89]:


b[:, 1]  # all rows, column 1


# In[90]:


b[1, :]


# In[91]:


b[1:2, :] #The first expression returns row 1 as a 1D array of shape (12,),
          #while the second returns that same row as a 2D array of shape (1, 12).


# In[92]:


b


# In[93]:


b[(0,2), 2:5]  # rows 0 and 2, columns 2 to 4 (5-1)


# In[94]:


b[:, (-1, 2, -1)]  # all rows, columns -1 (last), 2 and -1 (again, and in this order)


# In[95]:


b[(-1, 2, -1, 2), (5, 9, 1, 9)]  # returns a 1D array with b[-1, 5], b[2, 9], b[-1, 1] and b[2, 9] (again)


# In[96]:


c = b.reshape(4,2,6)
c


# In[97]:


c[2, 1, 4]  # matrix 2, row 1, col 4


# In[98]:


c[2, :, 3]  # matrix 2, all rows, col 3


# In[99]:


c[2, 1]  # Return matrix 2, row 1, all columns.  This is equivalent to c[2, 1, :]


# In[100]:


c[2, ...]  #  matrix 2, all rows, all columns.  This is equivalent to c[2, :, :]


# In[101]:


c[2, 1, ...]  # matrix 2, row 1, all columns.  This is equivalent to c[2, 1, :]


# In[102]:


c[2, ..., 3]  # matrix 2, all rows, column 3.  This is equivalent to c[2, :, 3]


# In[103]:


c[..., 3]  # all matrices, all rows, column 3.  This is equivalent to c[:, :, 3]


# In[104]:


b = np.arange(48).reshape(4, 12)
b


# In[105]:


rows_on = np.array([True, False, True, False])
b[rows_on, :]  # Rows 0 and 2, all columns. Equivalent to b[(0, 2), :]


# In[106]:


cols_on = np.array([False, True, False] * 4)
b[:, cols_on]  # All rows, columns 1, 4, 7 and 10


# In[107]:


[False, True, False] * 4


# In[108]:


b[np.ix_(rows_on, cols_on)]


# In[109]:


np.ix_(rows_on, cols_on)


# In[110]:


b


# In[111]:


b[b % 3 == 1]


# In[112]:


c = np.arange(24).reshape(2, 3, 4)  # A 3D array (composed of two 3x4 matrices)
c


# In[113]:


for m in c:
    print("Item:")
    print(m)


# In[114]:


for i in range(len(c)):  # Note that len(c) == c.shape[0]
    print("Item:")
    print(c[i])


# In[115]:


for i in c.flat:
    print("Item:", i)


# In[116]:


q1 = np.full((3,4), 1.0)
q1


# In[117]:


q2 = np.full((4,4), 2.0)
q2


# In[118]:


q3 = np.full((3,4), 3.0)
q3


# In[119]:


q4 = np.vstack((q1, q2, q3))
q4


# In[120]:


q4.shape


# In[121]:


q5 = np.hstack((q1, q3))
q5


# In[122]:


q5.shape


# In[123]:


try:
    q5 = np.hstack((q1, q2, q3))
except ValueError as e:
    print(e)


# In[124]:


q7 = np.concatenate((q1, q2, q3), axis=0)  # Equivalent to vstack
q7


# In[125]:


q7.shape


# In[126]:


q8 = np.stack((q1, q3))
q8


# In[127]:


q8.shape


# In[128]:


r = np.arange(24).reshape(6,4)
r


# In[129]:


r1, r2, r3 = np.vsplit(r, 3)
r1


# In[130]:


r2


# In[131]:


r3


# In[132]:


r4, r5 = np.hsplit(r, 2)
r4


# In[133]:


r5


# In[134]:


t = np.arange(24).reshape(4,2,3)
t


# In[135]:


t.shape


# In[136]:


t1 = t.transpose((1,2,0))
t1


# In[137]:


t1.shape


# In[138]:


t2 = t.transpose()  # equivalent to t.transpose((2, 1, 0))
t2


# In[139]:


t2.shape


# In[140]:


t3 = t.swapaxes(0,1)  # equivalent to t.transpose((1, 0, 2))
t3


# In[141]:


t3.shape


# In[142]:


m1 = np.arange(10).reshape(2,5)
m1


# In[143]:


m1.T


# In[144]:


m2 = np.arange(5)
m2


# In[145]:


m2.T


# In[146]:


m2r = m2.reshape(1,5)
m2r


# In[147]:


m2r.T


# In[148]:


n1 = np.arange(10).reshape(2, 5)
n1


# In[149]:


n2 = np.arange(15).reshape(5,3)
n2


# In[150]:


n1.dot(n2)


# In[151]:


import numpy.linalg as linalg

m3 = np.array([[1,2,3],[5,7,11],[21,29,31]])
m3


# In[152]:


linalg.inv(m3)


# In[153]:


linalg.pinv(m3) # both are same inv or pinv


# In[154]:


m3.dot(linalg.inv(m3))


# In[155]:


np.eye(3)


# In[156]:


m3


# In[157]:


q, r = linalg.qr(m3)
q


# In[158]:


r


# In[159]:


q.dot(r)  # q.r equals m3


# In[160]:


linalg.det(m3)  # Computes the matrix determinant


# In[161]:


m3


# In[162]:


eigenvalues, eigenvectors = linalg.eig(m3)
eigenvalues # λ


# In[163]:


eigenvectors # v


# In[164]:


m3.dot(eigenvectors) - eigenvalues * eigenvectors  # m3.v - λ*v = 0


# In[165]:


m4 = np.array([[1,0,0,0,2], [0,0,3,0,0], [0,0,0,0,0], [0,2,0,0,0]])
m4


# In[166]:


U, S_diag, V = linalg.svd(m4)
U


# In[167]:


S_diag


# In[168]:


S = np.zeros((4, 5))
S[np.diag_indices(4)] = S_diag
S  # Σ


# In[169]:


V


# In[170]:


U.dot(S).dot(V) # U.Σ.V == m4


# In[171]:


np.diag(m3)  # the values in the diagonal of m3 (top left to bottom right)


# In[172]:


np.trace(m3)  # equivalent to np.diag(m3).sum()


# In[173]:


coeffs  = np.array([[2, 6], [5, 3]])
depvars = np.array([6, -9])
solution = linalg.solve(coeffs, depvars)
solution


# In[174]:


coeffs.dot(solution), depvars  # yep, it's the same


# In[175]:


np.allclose(coeffs.dot(solution), depvars)


# In[176]:


import math
data = np.empty((768, 1024))
for y in range(768):
    for x in range(1024):
        data[y, x] = math.sin(x*y/40.5)  # BAD! Very inefficient.


# In[177]:


x_coords = np.arange(0, 1024)  # [0, 1, 2, ..., 1023]
y_coords = np.arange(0, 768)   # [0, 1, 2, ..., 767]
X, Y = np.meshgrid(x_coords, y_coords)
X


# In[178]:


Y


# In[179]:


data = np.sin(X*Y/40.5)


# In[180]:


data


# In[181]:


import matplotlib.pyplot as plt
import matplotlib.cm as cm
fig = plt.figure(1, figsize=(7, 6))
plt.imshow(data, cmap=cm.hot, interpolation="bicubic")
plt.show()


# In[182]:


a = np.random.rand(2,3)
a


# In[183]:


np.save("my_array", a)


# In[184]:


with open("my_array.npy", "rb") as f:
    content = f.read()

content


# In[185]:


a_loaded = np.load("my_array.npy")
a_loaded


# In[186]:


np.savetxt("my_array.csv", a)


# In[187]:


with open("my_array.csv", "rt") as f:
    print(f.read())


# In[188]:


np.savetxt("my_array.csv", a, delimiter=",")


# In[189]:


a_loaded = np.loadtxt("my_array.csv", delimiter=",")
a_loaded


# In[190]:


b = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
b


# In[191]:


np.savez("my_arrays", my_a=a, my_b=b)


# In[192]:


with open("my_arrays.npz", "rb") as f:
    content = f.read()

repr(content)[:180] + "[...]"


# In[193]:


my_arrays = np.load("my_arrays.npz")
my_arrays


# In[194]:


my_arrays.keys()


# In[195]:


my_arrays["my_a"]


# # refer below link for more 
# # https://numpy.org/doc/stable/reference/index.html

# In[ ]:




