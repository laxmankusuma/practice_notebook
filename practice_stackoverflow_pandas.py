#!/usr/bin/env python
# coding: utf-8

# # How to iterate over rows in a DataFrame in Pandas

# In[1]:


import pandas as pd

df = pd.DataFrame({'c1': [10, 11, 12], 'c2': [100, 110, 120]})
df


# In[2]:


for index, row in df.iterrows():
    print(row['c1'], row['c2'])


# In[3]:


for row in df.itertuples(index=True, name='Pandas'):
    print(row.c1, row.c2)


# In[4]:


import numpy

df = pd.DataFrame({'a': numpy.random.randn(1000), 'b': numpy.random.randn(1000),'N': numpy.random.randint(100, 1000, (1000)), 'x': 'x'})

df


# In[5]:


get_ipython().run_line_magic('timeit', '[row.a * 2 for idx, row in df.iterrows()]')
# => 10 loops, best of 3: 50.3 ms per loop

get_ipython().run_line_magic('timeit', '[row[1] * 2 for row in df.itertuples()]')
# => 1000 loops, best of 3: 541 µs per loop


# In[6]:


df = pd.DataFrame({'c1': [10, 11, 12], 'c2': [100, 110, 120]})

#You can use the df.iloc function as follows:
for i in range(0, len(df)):
    print(df.iloc[i]['c1'], df.iloc[i]['c2'])


# # How to select rows from a DataFrame based on column values

# In[7]:


import pandas as pd
import numpy as np
df = pd.DataFrame({'A': 'foo bar foo bar foo bar foo foo'.split(),
                   'B': 'one one two three two two one three'.split(),
                   'C': np.arange(8), 'D': np.arange(8) * 2})
print(df)
print(df.loc[df['A'] == 'foo'])


# In[8]:


print(df.loc[df['B'].isin(['one','three'])])


# In[9]:


# Note, however, that if you wish to do this many times, it is more efficient to make an index first, and then use df.loc
df = df.set_index(['B'])
print(df.loc['one'])


# In[10]:


# or, to include multiple values from the index use df.index.isin:

df.loc[df.index.isin(['one','two'])]


# In[11]:


import pandas as pd

# Create data set
d = {'foo':[100, 111, 222],
     'bar':[333, 444, 555]}
df = pd.DataFrame(d)

# Full dataframe:
df


# In[12]:


df[df.foo == 222]


# In[13]:


df[(df.foo == 222) | (df.bar == 444)]


# In[14]:


df = pd.DataFrame(np.random.rand(10, 3), columns=list('abc'))
df


# In[15]:


df[(df.a < df.b) & (df.b < df.c)]


# In[16]:


df.query('(a < b) & (b < c)')


# In[17]:


import pandas as pd
import numpy as np
df = pd.DataFrame({'A': 'foo bar foo bar foo bar foo foo'.split(),
                   'B': 'one one two three two two one three'.split(),
                   'C': np.arange(8), 'D': np.arange(8) * 2})
print(df)
print(df.loc[df['A'] == 'foo'])
df.iloc[np.where(df.A.values=='foo')]


# In[18]:


get_ipython().run_line_magic('timeit', "df.iloc[np.where(df.A.values=='foo')]  # fastest")
get_ipython().run_line_magic('timeit', "df.loc[df['A'] == 'foo']")
get_ipython().run_line_magic('timeit', "df.loc[df['A'].isin(['foo'])]")
get_ipython().run_line_magic('timeit', "df[df.A=='foo']")
get_ipython().run_line_magic('timeit', 'df.query(\'(A=="foo")\')  # slowest')


# # Renaming columns in pandas

# In[19]:


df = pd.DataFrame({'$a':[1,2], '$b': [10,20]})
df


# In[20]:


df.columns = ['a', 'b']
df


# In[21]:


df = df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'})
# Or rename the existing DataFrame (rather than creating a copy) 
df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'}, inplace=True)


# In[22]:


df = pd.DataFrame('x', index=range(3), columns=list('abcde'))
df


# In[23]:


df2 = df.rename({'a': 'X', 'b': 'Y'}, axis=1)  # new method
df2 = df.rename({'a': 'X', 'b': 'Y'}, axis='columns')
df2 = df.rename(columns={'a': 'X', 'b': 'Y'})  # old method  

df2


# df.rename({'a': 'X', 'b': 'Y'}, axis=1, inplace=True)
# df

# In[24]:


df2 = df.set_axis(['V', 'W', 'X', 'Y', 'Z'], axis=1, inplace=False)
df2


# In[25]:


df.columns = ['V', 'W', 'X', 'Y', 'Z']
df


# In[26]:


df.columns = df.columns.str.replace('$','')


# # Delete column from pandas DataFrame

# In[27]:


df = pd.DataFrame({'A': 'foo bar foo bar foo bar foo foo'.split(),
                   'E': 'foo bar foo bar foo bar foo foo'.split(),
                   'F': 'foo bar foo bar foo bar foo foo'.split(),
                   'B': 'one one two three two two one three'.split(),
                   'C': np.arange(8), 'D': np.arange(8) * 2})
df


# In[28]:


del df['A']
df


# In[29]:


df = df.drop('B', 1)
df


# In[30]:


df.drop('C', axis=1, inplace=True)
df


# In[31]:


df.drop(['E', 'F'], axis=1, inplace=True)
df


# In[32]:


df = pd.DataFrame({'A': 'foo bar foo bar foo bar foo foo'.split(),
                   'E': 'foo bar foo bar foo bar foo foo'.split(),
                   'F': 'foo bar foo bar foo bar foo foo'.split(),
                   'B': 'one one two three two two one three'.split(),
                   'C': np.arange(8), 'D': np.arange(8) * 2})
print(df)
#delete first, second, fourth
df.drop(df.columns[[0,1,3]], axis=1, inplace=True)
df


# # Selecting multiple columns in a pandas dataframe

# In[33]:


df = pd.DataFrame({'A': 'foo bar foo bar foo bar foo foo'.split(),
                   'E': 'foo bar foo bar foo bar foo foo'.split(),
                   'F': 'foo bar foo bar foo bar foo foo'.split(),
                   'B': 'one one two three two two one three'.split(),
                   'C': np.arange(8), 'D': np.arange(8) * 2})
df


# In[34]:


df[['A','B']]


# In[35]:


df.iloc[:, 0:2] # Remember that Python does not slice inclusive of the ending index.


# In[36]:


#see the comma(,) , before comma is row and after comma is column
df.iloc[0:2, : ]


# In[37]:


df1 = df.iloc[0, 0:2].copy() # To avoid the case where changing df1 also changes df
df1


# In[38]:


df.loc[:, 'E':'B']


# In[39]:


df.filter(['A', 'B'])


# # How do I get the row count of a pandas DataFrame?

# In[40]:


df


# In[41]:


df.shape


# In[42]:


len(df.index)


# In[43]:


df.count()


# In[44]:


print(len(df.columns))


# In[45]:


df.groupby('A').size()


# In[46]:


df.groupby('A').count()


# # Get list from pandas DataFrame column headers

# In[47]:


list(df.columns.values)


# In[48]:


list(df)


# In[49]:


df.columns.values.tolist()


# In[50]:


df.columns.tolist()


# In[51]:


get_ipython().run_line_magic('timeit', 'df.columns.tolist()')
get_ipython().run_line_magic('timeit', 'df.columns.values.tolist()')


# In[52]:


get_ipython().run_line_magic('timeit', '[column for column in df]')
get_ipython().run_line_magic('timeit', 'df.columns.values.tolist()')
get_ipython().run_line_magic('timeit', 'list(df)')
get_ipython().run_line_magic('timeit', 'list(df.columns.values)')


# In[53]:


[c for c in df]


# In[54]:


sorted(df)


# In[55]:


[*df]


# In[56]:


{*df}


# In[57]:


*df,


# In[58]:


*cols, = df
cols


# In[59]:


df.keys()


# # Adding new column to existing DataFrame in Python pandas

# In[60]:


df = pd.DataFrame({'A': 'foo bar foo bar foo bar foo foo'.split(),
                   'E': 'foo bar foo bar foo bar foo foo'.split(),
                   'F': 'foo bar foo bar foo bar foo foo'.split(),
                   'B': 'one one two three two two one three'.split(),
                   'C': np.arange(8), 'D': np.arange(8) * 2})
df


# In[61]:


sLength = len(df['A'])
df['X'] = pd.Series(np.random.randn(sLength), index=df.index)
df


# In[62]:


df['X1'] = "X1"
df


# In[63]:


df.loc[ : , 'new_col'] = "list_of_values"
df


# In[64]:


df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
df


# In[65]:


df.assign(mean_a=df.a.mean(), mean_b=df.b.mean())


# In[66]:


df['i'] = None
df


# # How to change the order of DataFrame columns?

# In[67]:


df


# In[68]:


df = df[['a', 'i', 'b']]


# In[69]:


df


# # Create pandas Dataframe by appending one row at a time

# In[70]:


import pandas as pd
from numpy.random import randint


# In[71]:


df = pd.DataFrame(columns=['lib', 'qty1', 'qty2'])


# In[72]:


for i in range(5):
    df.loc[i] = ['name' + str(i)] + list(randint(10, size=2))
                  
df


# In[73]:


import pandas as pd
import numpy as np
import time

# del df1, df2, df3, df4
numOfRows = 1000
# append
startTime = time.perf_counter()
df1 = pd.DataFrame(np.random.randint(100, size=(5,5)), columns=['A', 'B', 'C', 'D', 'E'])
for i in range( 1,numOfRows-4):
    df1 = df1.append( dict( (a,np.random.randint(100)) for a in ['A','B','C','D','E']), ignore_index=True)
print('Elapsed time: {:6.3f} seconds for {:d} rows'.format(time.perf_counter() - startTime, numOfRows))
print(df1.shape)

# .loc w/o prealloc
startTime = time.perf_counter()
df2 = pd.DataFrame(np.random.randint(100, size=(5,5)), columns=['A', 'B', 'C', 'D', 'E'])
for i in range( 1,numOfRows):
    df2.loc[i]  = np.random.randint(100, size=(1,5))[0]
print('Elapsed time: {:6.3f} seconds for {:d} rows'.format(time.perf_counter() - startTime, numOfRows))
print(df2.shape)

# .loc with prealloc
df3 = pd.DataFrame(index=np.arange(0, numOfRows), columns=['A', 'B', 'C', 'D', 'E'] )
startTime = time.perf_counter()
for i in range( 1,numOfRows):
    df3.loc[i]  = np.random.randint(100, size=(1,5))[0]
print('Elapsed time: {:6.3f} seconds for {:d} rows'.format(time.perf_counter() - startTime, numOfRows))
print(df3.shape)

# dict
startTime = time.perf_counter()
row_list = []
for i in range (0,5):
    row_list.append(dict( (a,np.random.randint(100)) for a in ['A','B','C','D','E']))
for i in range( 1,numOfRows-4):
    dict1 = dict( (a,np.random.randint(100)) for a in ['A','B','C','D','E'])
    row_list.append(dict1)

df4 = pd.DataFrame(row_list, columns=['A','B','C','D','E'])
print('Elapsed time: {:6.3f} seconds for {:d} rows'.format(time.perf_counter() - startTime, numOfRows))
print(df4.shape)


# # Change column type in pandas

# In[74]:


s = pd.Series(["8", 6, "7.5", 3, "0.9"]) # mixed string and numeric values
s


# In[75]:


pd.to_numeric(s) # convert everything to float values


# In[76]:


df = pd.DataFrame({'A': '1 2 3 4 5 6 7 8'.split(),
                   'E': 'foo bar foo bar foo bar foo foo'.split(),
                   'F': 'foo bar foo bar foo bar foo foo'.split(),
                   'B': 'one one two three two two one three'.split(),
                   'C': np.arange(8), 'D': np.arange(8) * 2})
df.info()


# In[77]:


df["A"] = pd.to_numeric(df["A"])
df.info()


# In[78]:


s = pd.Series(['1', '2', '4.7', 'pandas', '10'])
s


# In[79]:


pd.to_numeric(s, errors='coerce')


# In[80]:


pd.to_numeric(s, errors='ignore')


# In[81]:


df.apply(pd.to_numeric, errors='ignore')
df.info()


# In[82]:


s = pd.Series([1, 2, -7])
print(s)
print(pd.to_numeric(s, downcast='integer'))
print(pd.to_numeric(s, downcast='float'))


# In[83]:


print(df.info())
# convert all DataFrame columns to the int64 dtype
df = df.astype(str)
print(df.info())


# In[84]:


df = pd.DataFrame({'a': [7, 1, 5], 'b': ['3','2','1']}, dtype='object')
print(df.dtypes)
df = df.infer_objects()
print(df.dtypes)


# In[85]:


a = [['a', '1.2', '4.2'], ['b', '70', '0.03'], ['x', '5', '0']]
df = pd.DataFrame(a, columns=['one', 'two', 'three'])
print(df.dtypes)
df[['two', 'three']] = df[['two', 'three']].astype(float)
print(df.dtypes)


# # How to drop rows of Pandas DataFrame whose value in a certain column is NaN

# In[86]:


df = pd.DataFrame(np.random.randn(10,3))
df


# In[87]:


df.iloc[::2,0] = np.nan; df.iloc[::4,1] = np.nan; df.iloc[::3,2] = np.nan;
df


# In[88]:


df.dropna()     #drop all rows that have any NaN values


# In[89]:


df.dropna(how='all')     #drop only if ALL columns are NaN


# In[90]:


df.dropna(thresh=2)   #Drop row if it does not have at least two values that are **not** NaN


# In[91]:


df.dropna(subset=[1])   #Drop only if NaN in specific column (as asked in the question)


# In[92]:


df[pd.notnull(df[2])]


# In[93]:


df[df[0].notnull()]


# In[94]:


df[~df[0].notnull()]


# # Use a list of values to select rows from a pandas dataframe

# In[95]:


df = pd.DataFrame({'A': [5,6,3,4], 'B': [1,2,3,5]})
df


# In[96]:


df[df['A'].isin([3, 6])]


# In[97]:


df[~df['A'].isin([3, 6])]


# # How to deal with SettingWithCopyWarning in Pandas

# In[98]:


df


# In[99]:


df2 = df[['A']]
df2['A'] /= 2


# In[100]:


df2 = df.loc[:, ['A']]
df2['A'] /= 2     # Does not raise 
df2


# In[101]:


pd.options.mode.chained_assignment = None
df2['A'] /= 2
df2


# In[102]:


df2 = df[['A']].copy(deep=True)
df2['A'] /= 2
df2


# In[103]:


#dropping a column on the copy may affect the original
data1 = {'A': [111, 112, 113], 'B':[121, 122, 123]}
df1 = pd.DataFrame(data1)
df1


# In[104]:


df2 = df1


# In[105]:


df2.drop('A', axis=1, inplace=True)
df1


# In[106]:


#dropping a column on the original affects the copy
data1 = {'A': [111, 112, 113], 'B':[121, 122, 123]}
df1 = pd.DataFrame(data1)
df1


# In[107]:


df2 = df1
df2.drop('A', axis=1, inplace=True)
df1


# In[108]:


data1 = {'A': [111, 112, 113], 'B':[121, 122, 123]}
df1 = pd.DataFrame(data1)
df1


# In[109]:


import copy
df2 = copy.deepcopy(df1)
df2


# In[110]:


# Dropping a column on df1 does not affect df2
df2.drop('A', axis=1, inplace=True)
df1


# # Writing a pandas DataFrame to CSV file

# In[111]:


# REFER BELOW LINK FOR MORE INFO
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html#pandas.DataFrame.to_csv
# df.to_csv(file_name, sep='\t',encoding='utf-8', index=False,header = False, columns= ['x', 'y'], compression='gzip',
#          date_format='%d/%m/%Y', na_rep='N/A')
# df.to_csv (r'C:\Users\John\Desktop\export_dataframe.csv', index = None, header=True) 
# df.to_csv(r'./export/dftocsv.csv', sep='\t', encoding='utf-8', header='true')
# df.to_dense().to_csv("submission.csv", index = False, sep=',', encoding='utf-8')


# # Convert list of dictionaries to a pandas DataFrame

# In[112]:


d = [{'points': 50, 'time': '5:00', 'year': 2010}, 
{'points': 25, 'time': '6:00', 'month': "february"}, 
{'points':90, 'time': '9:00', 'month': 'january'}, 
{'points_h1':20, 'month': 'june'}]
d


# In[113]:


# The following methods all produce the same output.
print(pd.DataFrame(d))
print(pd.DataFrame.from_dict(d))
pd.DataFrame.from_records(d)


# In[114]:


data_c = [
 {'A': 5, 'B': 0, 'C': 3, 'D': 3},
 {'A': 7, 'B': 9, 'C': 3, 'D': 5},
 {'A': 2, 'B': 4, 'C': 7, 'D': 6}]
pd.DataFrame.from_dict(data_c, orient='columns')


# In[115]:


data_i ={
 0: {'A': 5, 'B': 0, 'C': 3, 'D': 3},
 1: {'A': 7, 'B': 9, 'C': 3, 'D': 5},
 2: {'A': 2, 'B': 4, 'C': 7, 'D': 6}}
pd.DataFrame.from_dict(data_i, orient='index')


# # Pretty-print an entire Pandas Series / DataFrame

# In[116]:


print(df)


# In[117]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)
# with option_context('display.max_rows', 10, 'display.max_columns', 5):
#     print(df)

# pd.set_option('display.height',1000)
# pd.set_option('display.max_rows',500)
# pd.set_option('display.max_columns',500)
# pd.set_option('display.width',1000)


# In[118]:


print(df.to_string())


# In[119]:


pd.describe_option('display')


# # How do I expand the output display to see more columns of a pandas DataFrame?

# In[120]:


# refer link below
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.set_option.html
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# # How are iloc and loc different?

# In[121]:


s = pd.Series(list("abcdef"), index=[49, 48, 47, 0, 1, 2])
s


# In[122]:


# for loc and iloc , hint is observe the comma(,). before comma is row and after the comma is column


# In[123]:


s.loc[0]    # value at index label 0


# In[124]:


s.iloc[0]   # value at index location 0


# In[125]:


s.loc[0:1]  # rows at index labels between 0 and 1 (inclusive)


# In[126]:


s.iloc[0:1] # rows at index location between 0 and 1 (exclusive)


# In[127]:


df = pd.DataFrame(np.arange(25).reshape(5, 5),  
                      index=list('abcde'), 
                      columns=['x','y','z', 8, 9])
df


# In[128]:


df.iloc[:df.index.get_loc('c') + 1, :4]


# In[129]:


df.index.get_loc('c') + 1


# # Deleting DataFrame row in Pandas based on column value

# In[130]:


df


# In[131]:


df2 = df[df.x != 0]
df2


# In[132]:


df2 = df[df.x != None] #Doesn't do anything:
df2


# In[133]:


df2 = df[df.x.notnull()]
df2


# In[134]:


df.drop(df.loc[df['x']==0].index, inplace=True)


# In[135]:


df


# In[136]:


df[(df.x != 0) & (df.x != 10)]


# In[137]:


df = df.drop(df[df['x']==0].index)
df


# # Combine two columns of text in pandas dataframe

# In[138]:


df = pd.DataFrame(np.arange(25).reshape(5, 5),  
                      index=list('abcde'), 
                      columns=['x','y','z', 8, 9])
df


# In[139]:


df["w"] = df["x"] + df["y"]
df


# In[140]:


d = [{'points': 50, 'time': '5:00', 'year': 2010}, 
{'points': 25, 'time': '6:00', 'month': "february"}, 
{'points':90, 'time': '9:00', 'month': 'january'}, 
{'points_h1':20, 'month': 'june'}]

df = pd.DataFrame(d)
df


# In[141]:


df.info()


# In[142]:


df["w"] = df["points"].astype(str) + df["time"].astype(str)
df


# In[143]:


df["w"] = df["points"] + df["year"]
df


# In[144]:


get_ipython().run_line_magic('timeit', "df['points'].astype(str) + df['time'].astype(str)")

get_ipython().run_line_magic('timeit', "df['points'].map(str) + df['time'].map(str)")

# %timeit df.points.str.cat(df.time.str)

get_ipython().run_line_magic('timeit', "df.loc[:, ['points','time']].astype(str).sum(axis=1)")

get_ipython().run_line_magic('timeit', "df[['points','time']].astype(str).sum(axis=1)")

get_ipython().run_line_magic('timeit', "df[['points','time']].apply(lambda x : '{}{}'.format(x[0],x[1]), axis=1)")


# # Creating an empty Pandas DataFrame, then filling it?

# In[145]:


import datetime
import pandas as pd
import numpy as np

todays_date = datetime.datetime.now().date()
todays_date


# In[146]:


index = pd.date_range(todays_date-datetime.timedelta(10), periods=10, freq='D')
index


# In[147]:


columns = ['A','B', 'C']


# In[148]:


df_ = pd.DataFrame(index=index, columns=columns)
df_


# In[149]:


df_ = df_.fillna(0) # with 0s rather than NaNs
df_


# In[150]:


data = np.array([np.arange(10)]*3).T
data


# In[151]:


df = pd.DataFrame(data, index=index, columns=columns)
df


# In[152]:


# Initialize empty frame with column names
col_names =  ['A', 'B', 'C']
my_df  = pd.DataFrame(columns = col_names)
my_df


# In[153]:


# Add a new record to a frame
my_df.loc[len(my_df)] = [2, 4, 5]
my_df


# In[154]:


# You also might want to pass a dictionary:
my_dic = {'A':2, 'B':4, 'C':5}
my_df.loc[len(my_df)] = my_dic 
my_df


# In[155]:


# Append another frame to your existing frame
col_names =  ['A', 'B', 'C']
my_df2  = pd.DataFrame(columns = col_names)
my_df = my_df.append(my_df2)
my_df


# # Set value for particular cell in pandas DataFrame using index

# In[156]:


df = pd.DataFrame(index=['A','B','C'], columns=['x','y'])
df


# In[157]:


df.at['C', 'x'] = 10
df


# In[158]:


df.set_value('C', 'x', 10)
df


# In[159]:


df['x']['C'] = 10
df


# In[160]:


get_ipython().run_line_magic('timeit', "df.set_value('C', 'x', 10)")
get_ipython().run_line_magic('timeit', "df['x']['C'] = 10")
get_ipython().run_line_magic('timeit', "df.at['C', 'x'] = 10")


# # How to count the NaN values in a column in pandas DataFrame

# In[161]:


s = pd.Series([1,2,3, np.nan, np.nan])
s


# In[162]:


s.isna().sum()   # or s.isnull().sum() for older pandas versions


# In[163]:


df = pd.DataFrame({'a':[1,2,np.nan], 'b':[np.nan,1,np.nan]})
df


# In[164]:


df.isna().sum()


# In[165]:


count_nan = len(df) - df.count()
count_nan


# In[166]:


df.isnull().sum(axis = 0) # This will give number of NaN values in every column.


# In[167]:


df.isnull().sum(axis = 1) # If you need, NaN values in every row,


# In[168]:


df = pd.DataFrame({'a':[1,2,np.nan], 'b':[np.nan,1,np.nan]})
df


# In[169]:


for col in df:
    print(df[col].value_counts(dropna=False))


# In[170]:


df


# In[171]:


df.isnull().sum().sort_values(ascending = False)


# In[172]:


df.isnull().sum().sort_values(ascending = False).head(15) # The below will print first 15 Nan columns in descending order.       


# In[173]:


df.isnull().any().any()


# In[174]:


df.isnull().values.sum()


# In[175]:


df.isnull().any()


# In[176]:


df.a.isnull().sum()


# In[177]:


df.b.isnull().sum()


# # Select by partial string from a pandas DataFrame

# In[178]:


d = [{'points': 50, 'time': '5:00', 'year': 2010}, 
{'points': 25, 'time': '6:00', 'month': "february"}, 
{'points':90, 'time': '9:00', 'month': 'january'}, 
{'points_h1':20, 'month': 'june'}]

df = pd.DataFrame(d)
df


# In[179]:


df[df['month'].str.contains("janua",na=False)]


# In[180]:


df[df['month'].str.contains("janua|ne",na=False)]


# In[181]:


df[df['month'].str.contains(".*uary",na=False)]


# In[182]:


get_ipython().run_line_magic('timeit', "df[df['month'].str.contains('uary',na=False)]")
get_ipython().run_line_magic('timeit', "df[df['month'].str.contains('uary', na=False,regex=False)]")


# In[183]:


s = pd.Series(['foo', 'foobar', np.nan, 'bar', 'baz', 123])
s


# In[184]:


s.str.contains('foo|bar')


# In[185]:


s.str.contains('foo|bar', na=False)


# In[186]:


df


# In[187]:


df.filter(like='mon')  # select columns which contain the word mon


# # How to convert index of a pandas dataframe into a column?

# In[188]:


d = [{'points': 50, 'time': '5:00', 'year': 2010}, 
{'points': 25, 'time': '6:00', 'month': "february"}, 
{'points':90, 'time': '9:00', 'month': 'january'}, 
{'points_h1':20, 'month': 'june'}]

df = pd.DataFrame(d)
df


# In[189]:


df = df.rename_axis('index1').reset_index()
df


# In[190]:


df['index1'] = df.index
df


# In[191]:


# If you want to use the reset_index method and also preserve your existing index you should use:
df.reset_index().set_index('index', drop=False)


# # Converting a Pandas GroupBy output from Series to DataFrame

# In[192]:


df1 = pd.DataFrame( { 
    "Name" : ["Alice", "Bob", "Mallory", "Mallory", "Bob" , "Mallory"] , 
    "City" : ["Seattle", "Seattle", "Portland", "Seattle", "Seattle", "Portland"] } )
df1


# In[193]:


g1 = df1.groupby( [ "Name", "City"] ).count()
g1


# In[194]:


type(g1)


# In[195]:


g1.index


# In[196]:


g1.add_suffix('_Count').reset_index()


# In[197]:


pd.DataFrame({'count' : df1.groupby( [ "Name", "City"] ).size()})


# # Convert pandas dataframe to NumPy array

# In[198]:


import numpy as np
import pandas as pd

index = [1, 2, 3, 4, 5, 6, 7]
a = [np.nan, np.nan, np.nan, 0.1, 0.1, 0.1, 0.1]
b = [0.2, np.nan, 0.2, 0.2, 0.2, np.nan, np.nan]
c = [np.nan, 0.5, 0.5, np.nan, 0.5, 0.5, np.nan]
df = pd.DataFrame({'A': a, 'B': b, 'C': c}, index=index)
df = df.rename_axis('ID')
df


# In[199]:


df.values


# In[200]:


df.to_numpy() # Convert the entire DataFrame


# In[201]:


df[['A', 'C']].to_numpy() # Convert specific columns


# In[202]:


df.to_records() # If you need the dtypes in the result...


# # How to filter Pandas dataframe using 'in' and 'not in' like in SQL

# In[203]:


df1 = pd.DataFrame( { 
    "Name" : ["Alice", "Bob", "Mallory", "Mallory", "Bob" , "Mallory"] , 
    "City" : ["Seattle", "hyderabad", "Portland", "Seattle", "Seattle", "Portland"] } )
df1


# In[204]:


df1[df1.City.isin(['hyderabad','Portland'])]


# In[205]:


df1[~df1.City.isin(['hyderabad','Portland'])]


# In[206]:


df1.query("City in ('hyderabad','Portland')")


# In[207]:


df1.query("City not in ('hyderabad','Portland')")


# In[208]:


df = pd.DataFrame({'countries': ['US', 'UK', 'Germany', np.nan, 'China']})
df


# In[209]:


c1 = ['UK', 'China']             # list
c2 = {'Germany'}                 # set
c3 = pd.Series(['China', 'US'])  # Series
c4 = np.array(['US', 'UK'])      # array


# In[210]:


df[df['countries'].isin(c1)]


# In[211]:


df[df['countries'].isin(c2)]


# In[212]:


df[df['countries'].isin(c3)]


# In[213]:


df[df['countries'].isin(c4)]


# In[214]:


df2 = pd.DataFrame({
    'A': ['x', 'y', 'z', 'q'], 'B': ['w', 'a', np.nan, 'x'], 'C': np.arange(4)})
df2


# # Shuffle DataFrame rows

# In[215]:


df


# In[216]:


df.sample(frac=1)


# In[217]:


df = df.sample(frac=1).reset_index(drop=True)
df


# In[218]:


from sklearn.utils import shuffle
df = shuffle(df)
df


# # Get statistics for each group (such as count, mean, etc) using pandas GroupBy?

# In[219]:


data_i ={
 0: {'A': 5, 'B': 0, 'C': 3, 'D': 3},
 1: {'A': 7, 'B': 9, 'C': 3, 'D': 3},
 2: {'A': 2, 'B': 4, 'C': 7, 'D': 6}}
df = pd.DataFrame.from_dict(data_i, orient='index')
df


# In[220]:


df[['A', 'B', 'C', 'D']].groupby(['C', 'D']).agg(['mean', 'count'])


# In[221]:


df.groupby(['C','D']).size()


# In[222]:


# Usually you want this result as a DataFrame (instead of a Series) so you can do:
df.groupby(['C','D']).size().reset_index(name='counts')


# In[223]:


df.groupby(['C','D'])['A'].describe()


# In[224]:


df[['A','B','C','D']].groupby(['C','D']).count().reset_index()


# # How to replace NaN values by Zeroes in a column of a Pandas Dataframe?

# In[225]:


df


# In[226]:


import numpy as np
import pandas as pd

index = [1, 2, 3, 4, 5, 6, 7]
a = [np.nan, np.nan, np.nan, 0.1, 0.1, 0.1, 0.1]
b = [0.2, np.nan, 0.2, 0.2, 0.2, np.nan, np.nan]
c = [np.nan, 0.5, 0.5, np.nan, 0.5, 0.5, np.nan]
df = pd.DataFrame({'A': a, 'B': b, 'C': c}, index=index)
df = df.rename_axis('ID')
df


# In[227]:


df.fillna(0, inplace=True)
df


# # Difference between map, applymap and apply methods in Pandas

# In[228]:


frame = pd.DataFrame(np.random.randn(4, 3), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
frame


# In[229]:


f = lambda x: x.max() - x.min()


# In[230]:


frame.apply(f)


# In[231]:


format = lambda x: '%.2f' % x


# In[232]:


frame.applymap(format)


# In[233]:


frame['e'].map(format)


# In[234]:


'''
map is defined on Series ONLY
applymap is defined on DataFrames ONLY
apply is defined on BOTH
'''


# # UnicodeDecodeError when reading CSV file in Pandas with Python

# read_csv takes an encoding option to deal with files in different formats. I mostly use read_csv('file', encoding = "ISO-8859-1"), or alternatively encoding = "utf-8" for reading, and generally utf-8 for to_csv
# 
# pd.read_csv('immigration.csv', encoding = "ISO-8859-1", engine='python')

# # Pandas Merging 101

# In[235]:


np.random.seed(0)
left = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'value': np.random.randn(4)})    
right = pd.DataFrame({'key': ['B', 'D', 'E', 'F'], 'value': np.random.randn(4)})
left


# In[236]:


right


# In[237]:


left.merge(right, on='key')


# In[238]:


left.merge(right, on='key', how='left')


# In[239]:


left.merge(right, on='key', how='right')


# In[240]:


left.merge(right, on='key', how='outer')


# In[241]:


(left.merge(right, on='key', how='left', indicator=True)
     .query('_merge == "left_only"')
     .drop('_merge', 1))


# In[242]:


left.merge(right, on='key', how='left', indicator=True)


# In[243]:


(left.merge(right, on='key', how='right', indicator=True)
     .query('_merge == "right_only"')
     .drop('_merge', 1))


# In[244]:


(left.merge(right, on='key', how='outer', indicator=True)
     .query('_merge != "both"')
     .drop('_merge', 1))


# In[245]:


#Different names for key columns
left2 = left.rename({'key':'keyLeft'}, axis=1)
right2 = right.rename({'key':'keyRight'}, axis=1)
left2


# In[246]:


right2


# In[247]:


left2.merge(right2, left_on='keyLeft', right_on='keyRight', how='inner')


# In[248]:


#Avoiding duplicate key column in output
left3 = left2.set_index('keyLeft')
left3.merge(right2, left_index=True, right_on='keyRight')


# In[249]:


#Merging on multiple columns
# left.merge(right, on=['key1', 'key2'] ...)
# Or, in the event the names are different,
# left.merge(right, left_on=['lkey1', 'lkey2'], right_on=['rkey1', 'rkey2'])


# # Import multiple csv files into pandas and concatenate into one DataFrame

# In[250]:


'''
import pandas as pd
import glob

path = r'C:\DRO\DCL_rawdata_files' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)
'''


# In[251]:


'''path = r'C:\DRO\DCL_rawdata_files'                     # use your path
all_files = glob.glob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent

df_from_each_file = (pd.read_csv(f) for f in all_files)
concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)
# doesn't create a list, nor does it append to one'''


# In[252]:


'''import glob
import os
import pandas as pd   
df = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', "my_files*.csv"))))'''


# # How to avoid Python/Pandas creating an index in a saved csv?

# In[253]:


# df.to_csv('your.csv', index=False)


# # Filter dataframe rows if value in column is in a set list of values [duplicate]

# In[254]:


# b = df[(df['a'] > 1) & (df['a'] < 5)]


# # Truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all()

# In[255]:


x = pd.Series([])
x.empty


# In[256]:


x = pd.Series([1])
x.empty


# In[257]:


x = pd.Series([100])
x


# In[258]:


(x > 50).bool()


# In[259]:


(x < 50).bool()


# In[260]:


x = pd.Series([100])
x.item()


# In[261]:


x = pd.Series([0, 1, 2])
x.all()   # because one element is zero


# In[262]:


x.any()   # because one (or more) elements are non-zero


# # How to apply a function to two columns of Pandas dataframe

# In[263]:


import pandas as pd

df = pd.DataFrame({'ID':['1', '2', '3'], 'col_1': [0, 2, 3], 'col_2':[1, 4, 5]})
mylist = ['a', 'b', 'c', 'd', 'e', 'f']

def get_sublist(sta,end):
    return mylist[sta:end+1]

df['col_3'] = df.apply(lambda x: get_sublist(x.col_1, x.col_2), axis=1)
df


# # How to get a value from a cell of a dataframe?

# In[264]:


df


# In[265]:


df.iloc[0]


# In[266]:


df.iloc[0]['col_2']


# # Selecting a row of pandas series/dataframe by integer index

# In[267]:


df = pd.DataFrame(np.random.rand(5,2),index=range(0,10,2),columns=list('AB'))
df


# In[268]:


df.iloc[[2]]


# In[269]:


df.loc[[2]]


# # How to pivot a dataframe?

# In[270]:


import numpy as np
import pandas as pd
from numpy.core.defchararray import add

np.random.seed([3,1415])
n = 20

cols = np.array(['key', 'row', 'item', 'col'])
arr1 = (np.random.randint(5, size=(n, 4)) // [2, 1, 2, 1]).astype(str)

df = pd.DataFrame(
    add(cols, arr1), columns=cols
).join(
    pd.DataFrame(np.random.rand(n, 2).round(2)).add_prefix('val')
)

df


# In[271]:


df.duplicated(['row', 'col']).any()


# In[272]:


df.pivot_table(
    values='val0', index='row', columns='col',
    fill_value=0, aggfunc='mean')


# In[273]:


df.groupby(['row', 'col'])['val0'].mean().unstack(fill_value=0)


# In[274]:


pd.crosstab(
    index=df['row'], columns=df['col'],
    values=df['val0'], aggfunc='mean').fillna(0)


# In[275]:


df.pivot_table(
    values='val0', index='row', columns='col',
    fill_value=0, aggfunc='sum')


# In[276]:


df.groupby(['row', 'col'])['val0'].sum().unstack(fill_value=0)


# In[277]:


pd.crosstab(
    index=df['row'], columns=df['col'],
    values=df['val0'], aggfunc='sum').fillna(0)


# In[278]:


df.pivot_table(
    values='val0', index='row', columns='col',
    fill_value=0, aggfunc=[np.size, np.mean])


# In[279]:


df.groupby(['row', 'col'])['val0'].agg(['size', 'mean']).unstack(fill_value=0)


# In[280]:


pd.crosstab(
    index=df['row'], columns=df['col'],
    values=df['val0'], aggfunc=[np.size, np.mean]).fillna(0, downcast='infer')


# In[281]:


df.pivot_table(
    values=['val0', 'val1'], index='row', columns='col',
    fill_value=0, aggfunc='mean')


# In[282]:


df.pivot_table(
    values='val0', index='row', columns=['item', 'col'],
    fill_value=0, aggfunc='mean')


# In[283]:


df.groupby(
    ['row', 'item', 'col']
)['val0'].mean().unstack(['item', 'col']).fillna(0).sort_index(1)


# In[284]:


df.pivot_table(
    values='val0', index=['key', 'row'], columns=['item', 'col'],
    fill_value=0, aggfunc='mean')


# In[285]:


df.groupby(
    ['key', 'row', 'item', 'col']
)['val0'].mean().unstack(['item', 'col']).fillna(0).sort_index(1)


# In[286]:


df.set_index(
    ['key', 'row', 'item', 'col']
)['val0'].unstack(['item', 'col']).fillna(0).sort_index(1)


# In[287]:


df.pivot_table(index='row', columns='col', fill_value=0, aggfunc='size')


# In[288]:


df.groupby(['row', 'col'])['val0'].size().unstack(fill_value=0)


# In[289]:


pd.crosstab(df['row'], df['col'])


# In[290]:


# get integer factorization `i` and unique values `r`
# for column `'row'`
i, r = pd.factorize(df['row'].values)
# get integer factorization `j` and unique values `c`
# for column `'col'`
j, c = pd.factorize(df['col'].values)
# `n` will be the number of rows
# `m` will be the number of columns
n, m = r.size, c.size
# `i * m + j` is a clever way of counting the 
# factorization bins assuming a flat array of length
# `n * m`.  Which is why we subsequently reshape as `(n, m)`
b = np.bincount(i * m + j, minlength=n * m).reshape(n, m)
# BTW, whenever I read this, I think 'Bean, Rice, and Cheese'
pd.DataFrame(b, r, c)


# In[291]:


pd.get_dummies(df['row']).T.dot(pd.get_dummies(df['col']))


# In[292]:


df.columns = df.columns.map('|'.join)
df


# In[293]:


df.columns = df.columns.map('{0[0]}|{0[1]}'.format) 

df


# In[294]:


d = data = {'A': {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 5},
 'B': {0: 'a', 1: 'b', 2: 'c', 3: 'a', 4: 'b', 5: 'a', 6: 'c'}}
df = pd.DataFrame(d)
df


# # Python Pandas Error tokenizing data

# data = pd.read_csv('file1.csv', error_bad_lines=False)

# # Remap values in pandas column with a dict

# In[295]:


df = pd.DataFrame({'col2': {0: 'a', 1: 2, 2: np.nan}, 'col1': {0: 'w', 1: 1, 2: 2}})
df


# In[296]:


di = {1: "A", 2: "B"}
df.replace({"col1": di})


# # Pandas read_csv low_memory and dtype options

# dashboard_df = pd.read_csv(p_file, sep=',', error_bad_lines=False, index_col=False, dtype='unicode') 
# 
# df = pd.read_csv('somefile.csv', low_memory=False)

# # Pandas - How to flatten a hierarchical index in columns

# In[297]:


df = pd.DataFrame({'col2': {0: 'a', 1: 2, 2: np.nan}, 'col1': {0: 'w', 1: 1, 2: 2}})
df


# In[298]:


df.columns.get_level_values(0)


# In[299]:


[''.join(col).strip() for col in df.columns.values]


# In[300]:


df.columns.map(''.join).str.strip()


# # How do I create test and train samples from one dataframe with pandas?

# In[301]:


df = pd.DataFrame(np.random.randn(100, 2))
df


# In[302]:


msk = np.random.rand(len(df)) < 0.8


# In[303]:


train = df[msk]
test = df[~msk]


# In[304]:


print(len(train))
print(len(test))
print(len(msk))


# In[305]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2)
print(df.shape[0])
print(train.shape[0])
print(test.shape[0])


# In[306]:


train = df.sample(frac=0.8,random_state=200) #random state is a seed value
test = df.drop(train.index)
print(df.shape[0])
print(train.shape[0])
print(test.shape[0])


# # Selecting/excluding sets of columns in pandas [duplicate]

# In[307]:


df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
df


# In[308]:


df.drop(df.columns[[1, 2]], axis=1)


# In[309]:


import numpy as np
import pandas as pd

# Create a dataframe with columns A,B,C and D
df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))

# include the columns you want
df[df.columns[df.columns.isin(['A', 'B'])]]


# In[310]:


# or more simply include columns:
df[['A', 'B']]


# In[311]:


# exclude columns you don't want
df[df.columns[~df.columns.isin(['C','D'])]]


# In[312]:


# or even simpler since 0.24
# with the caveat that it reorders columns alphabetically 
df[df.columns.difference(['C', 'D'])]


# # How to check whether a pandas DataFrame is empty?

# In[313]:


df


# In[314]:


df.empty


# In[315]:


len(df) == 0


# In[316]:


len(df.index) == 0


# # Pandas - Get first row value of a given column

# In[317]:


df


# In[318]:


df['A'].iloc[0]


# # How to store a dataframe using Pandas

# In[319]:


df


# In[320]:


df.to_pickle("file_name.pkl")  # where to save it, usually as a .pkl


# In[321]:


# Then you can load it back using:
df = pd.read_pickle("file_name.pkl")
df


# In[322]:


# Another popular choice is to use HDF5 (pytables) which offers very fast access times for large datasets:
import pandas as pd
store = pd.HDFStore('store.h5')

store['df'] = df  # save it
store['df']  # load it


# pickle: original ASCII data format
# 
# cPickle, a C library
# 
# pickle-p2: uses the newer binary format
# 
# json: standardlib json library
# 
# json-no-index: like json, but without index
# 
# msgpack: binary JSON alternative
# 
# CSV
# 
# hdfstore: HDF5 storage format

# # Pandas conditional creation of a series/dataframe column

# In[323]:


df


# In[324]:


import pandas as pd
import numpy as np

df = pd.DataFrame({'Type':list('ABBC'), 'Set':list('ZZXY')})
df


# In[325]:


df['color'] = np.where(df['Set']=='Z', 'green', 'red')
print(df)


# In[326]:


conditions = [
    (df['Set'] == 'Z') & (df['Type'] == 'A'),
    (df['Set'] == 'Z') & (df['Type'] == 'B'),
    (df['Type'] == 'B')]
choices = ['yellow', 'blue', 'purple']
df['color'] = np.select(conditions, choices, default='black')
print(df)


# In[327]:


df['color'] = ['red' if x == 'Z' else 'green' for x in df['Set']]
df


# In[328]:


import pandas as pd
import numpy as np

df = pd.DataFrame({'Type':list('ABBC'), 'Set':list('ZZXY')})
get_ipython().run_line_magic('timeit', "df['color'] = ['red' if x == 'Z' else 'green' for x in df['Set']]")
get_ipython().run_line_magic('timeit', "df['color'] = np.where(df['Set']=='Z', 'green', 'red')")
get_ipython().run_line_magic('timeit', "df['color'] = df.Set.map( lambda x: 'red' if x == 'Z' else 'green')")


# In[329]:


def set_color(row):
    if row["Set"] == "Z":
        return "red"
    elif row["Type"] == "C":
        return "blue"
    else:
        return "green"

df = df.assign(color=df.apply(set_color, axis=1))

print(df)


# # pandas: filter rows of DataFrame with operator chaining

# In[330]:


df = pd.DataFrame(np.random.randn(30, 3), columns=['a','b','c'])
df


# In[331]:


df.query('a > 0').query('0 < b < 0.5')


# In[332]:


df.query('a > 0 and 0 < b < 0.5')


# # Count the frequency that a value occurs in a dataframe column

# In[333]:


df = pd.DataFrame({'col':list('abssbab')})
df


# In[334]:


df['col'].value_counts()


# In[335]:


df.groupby('col').count()


# In[336]:


df['freq']=df.groupby('col')['col'].transform('count')
df


# In[337]:


df = pd.DataFrame({'col':list('abssbab')})
df.apply(pd.value_counts)


# In[338]:


df.apply(pd.value_counts).fillna(0)


# # How to select all columns, except one column in pandas?

# In[339]:


import pandas
import numpy as np
df = pd.DataFrame(np.random.rand(4,4), columns = list('abcd'))
df


# In[340]:


df.loc[:, df.columns != 'b']


# In[341]:


df.drop('b', axis=1)


# In[342]:


df = pd.DataFrame(np.random.rand(4,4), columns = list('abcd'))
df[df.columns.difference(['b'])]


# In[343]:


df.loc[:, ~df.columns.isin(['a', 'b'])]


# # How to group dataframe rows into list in pandas groupby

# In[344]:


df = pd.DataFrame( {'a':['A','A','B','B','B','C'], 'b':[1,2,5,5,4,6]})
df


# In[345]:


df.groupby('a')['b'].apply(list)


# # Convert Python dict into a dataframe

# In[346]:


d = {u'2012-06-08': 388,
 u'2012-06-09': 388,
 u'2012-06-10': 388,
 u'2012-06-11': 389,
 u'2012-06-12': 389,
 u'2012-06-13': 389,
 u'2012-06-14': 389,
 u'2012-06-15': 389,
 u'2012-06-16': 389,
 u'2012-06-17': 389,
 u'2012-06-18': 390,
 u'2012-06-19': 390,
 u'2012-06-20': 390,
 u'2012-06-21': 390,
 u'2012-06-22': 390,
 u'2012-06-23': 390,
 u'2012-06-24': 390,
 u'2012-06-25': 391,
 u'2012-06-26': 391,
 u'2012-06-27': 391,
 u'2012-06-28': 391,
 u'2012-06-29': 391,
 u'2012-06-30': 391,
 u'2012-07-01': 391,
 u'2012-07-02': 392,
 u'2012-07-03': 392,
 u'2012-07-04': 392,
 u'2012-07-05': 392,
 u'2012-07-06': 392}


# In[347]:


df = pd.DataFrame(d.items())
df


# In[348]:


pd.DataFrame(d.items(), columns=['Date', 'DateValue'])


# # How to check if a column exists in Pandas

# In[349]:


df = pd.DataFrame( {'a':['A','A','B','B','B','C'], 'b':[1,2,5,5,4,6]})
df


# In[350]:


'a' in df


# In[351]:


'a' in df.columns


# In[352]:


all([item in df.columns for item in ['a','b']])


# In[353]:


all([item in df.columns for item in ['a','c']])


# In[354]:


{'a', 'b'}.issubset(df.columns)


# In[355]:


{'a', 'c'}.issubset(df.columns)


# # What is the most efficient way to loop through dataframes with pandas?

# In[356]:


'''for index, row in df.iterrows():

    # do some logic here'''


# # Get list from pandas dataframe column or row?

# In[357]:


import pandas as pd

data_dict = {'one': pd.Series([1, 2, 3], index=['a', 'b', 'c']),
             'two': pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(data_dict)
print(f"DataFrame:\n{df}\n")
print(f"column types:\n{df.dtypes}")

col_one_list = df['one'].tolist()

col_one_arr = df['one'].to_numpy()

print(f"\ncol_one_list:\n{col_one_list}\ntype:{type(col_one_list)}")
print(f"\ncol_one_arr:\n{col_one_arr}\ntype:{type(col_one_arr)}")


# In[ ]:




