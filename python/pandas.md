#### demo1

```python
import pandas as pd
import numpy as np
s=pd.Series([1,3,6,np.nan,44,1])
print(s) # 输出的时候有一个序号

dates=pd.date_range('20230423',periods=6)
print(dates)

# dataframe 类似一个大的matric，类似于一个二维的numpy，第一个是数据，第二个是行索引，第三个是列索引
df=pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','h'])
print(df)

# 默认的索引是数字
df1=pd.DataFrame(np.arange(12).reshape((3,4)))
print(df1)

print(df1.dtypes)
print(df1.index)
print(df1.columns)

# describe()只是能够运算我们数字形式的一些属性，count，mean min
print(df1.describe())

# 矩阵转置
print(df1.T)

# 排序，是对列排序，倒序
print(df1.sort_index(axis=1,ascending=False))
print(df1.sort_index(axis=0,ascending=False))

# 对某一列进行排序
print(df1.sort_values(by=2))
print(df.sort_values(by='c'))




```



#### 选择数据

```python
"""
选择数据
"""

import pandas as pd
import numpy as np

dates=pd.date_range('20230423',periods=6)
df=pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=["A","B","C","D"])

print(df)

# 选择数据的某一列,比如打印A这一列，两个效果是一样的
# print(df['A'],df.A)

# 切片来进行选择行
print(df[0:3],df['20230424':'20230425'])

# select by label:loc

print(df.loc['20230423']) # 筛选某一行

print(df.loc[:,['A','B']]) # 行的数据保留，筛选列A，B所有行的数据

print(df.loc['20230423',["A","B"]]) # 指定行，指定列

# select by position：iloc
print(df.iloc[3:5,1:3])

# 一个一个的筛选,将1,3,5行的第一列和第二列筛选出来
print(df.iloc[[1,3,5],1:3])

# 将标签和位置一起混合筛选
# mixed selection：ix
# print(df.ix[:3,['A','C']]) # 行以数字进行筛选，列以标签数字进行筛选

# boolean indexing
print(df)

# 筛选df，但是只筛选df.A>8的df
print(df[df.A>8])

```



#### 设置值

```python
import pandas as pd
import numpy as np

dates=pd.date_range('20230423',periods=6)
df=pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=["A","B","C","D"])

# ，定位位置进行修改，修改第二行第二列的值，下标从0开始
df.iloc[2,2]=1111
print(df)

# 定位标签进行修改
df.loc['20230423','B']=2022
print(df)

# 修改df.A中数据,只修改这一列大于4的数据
# df.A[df.A>4]=0
# print(df)

# 当然也可以修改B这一列的
df.B[df.A>4]=0
print(df)


# 加入空列
df["F"]=np.nan
print(df)

# 加入一个新的序列,对上原先的行
df['E']=pd.Series([1,2,3,4,5,6],index=pd.date_range('20230423',periods=6))
print(df)
```



#### 数据连接

```python
import pandas as pd
import numpy as np

# df1=pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'])
# df2=pd.DataFrame(np.ones((3,4))*1,columns=['a','b','c','d'])
# df3=pd.DataFrame(np.ones((3,4))*2,columns=['a','b','c','d'])
# print(df1)
# print(df2)
# print(df3)
#
# res=pd.concat([df1,df2,df3],axis=0)
# res=pd.concat([df1,df2,df3],axis=0,ignore_index=True)
# print(res)


# join,可以将不一样的功能很好的处理一下

df1=pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'])
df2=pd.DataFrame(np.ones((3,4))*1,columns=['a','b','c','d'])

print(df1)
print(df2)

# res=pd.concat([df1,df2],join='inner')
# print(res)

# res=pd.concat([df1,df2],axis=1,join_axes=[df1.index])

# append


s1=pd.Series([1,2,3,4],index=['a','b','c','d'])
res=df1.append(s1,ignore_index=True)
print(res)
```



#### 处理缺失值

```python
import pandas as pd
import numpy as np

# df1=pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'])
# df2=pd.DataFrame(np.ones((3,4))*1,columns=['a','b','c','d'])
# df3=pd.DataFrame(np.ones((3,4))*2,columns=['a','b','c','d'])
# print(df1)
# print(df2)
# print(df3)
#
# res=pd.concat([df1,df2,df3],axis=0)
# res=pd.concat([df1,df2,df3],axis=0,ignore_index=True)
# print(res)


# join,可以将不一样的功能很好的处理一下

df1=pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'])
df2=pd.DataFrame(np.ones((3,4))*1,columns=['a','b','c','d'])

print(df1)
print(df2)

# res=pd.concat([df1,df2],join='inner')
# print(res)

# res=pd.concat([df1,df2],axis=1,join_axes=[df1.index])

# append


s1=pd.Series([1,2,3,4],index=['a','b','c','d'])
res=df1.append(s1,ignore_index=True)
print(res)
```



#### 绘图

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# plot data


# Series
data=pd.Series(np.random.randn(1000),index=np.arange(1000))
# 累加
data=data.cumsum()

# DataFrame
data=pd.DataFrame(np.random.randn(1000,4),index=np.arange(1000),columns=list('ABCD'))
data=data.cumsum()
print(data)


# 此时会绘出4组数据
# data.plot()

# plot methods:
# bar,hist,box,kdde,area,scatter,hexbin,pie

# 在一张图上，打印两组数据
ax=data.plot.scatter(x='A',y='B',color='DarkBlue',label='class one')
data.plot.scatter(x='A',y='C',color='DarkGreen',label='class two',ax=ax)
data.plot.scatter(x='A',y='D',color='DarkRed',label='class three',ax=ax)
# plt.legend()
plt.show()


```

