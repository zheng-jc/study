##### 1.numpy基础

```python
import numpy as np
# 首先定义一个数组，然后将数组转为numpy
arr=[[1,2,3],
     [2,3,4]]
array=np.array(arr)

print(array)
print("number of dim:",array.ndim) # 输出维度
print("shape:",array.shape) # 输出形状
print("size:",array.size) # 输出元素个数

a=np.array([2,3,4],dtype=np.int64)
print(a)
print(a.dtype)

# 创建一个5*5大小的全0的矩阵
b=np.zeros((5,5))
print(b)

# 创建一个6*5大小的全为1的矩阵
c=np.ones((6,5))
print(c)

# 输出3*4大小的很接近0的数
d=np.empty((3,4))# 输出很接近0的数
print(d)

# 输出输出10~20，不包括20，步伐为2依次递增
e=np.arange(10,20,2) # [10 12 14 16 18]
print(e)

# 输出0-11这12个数，并变换为一个3行4列的形状
f=np.arange(12).reshape((3,4))
print(f)


# 输出从1到10，把1~10分成6段[[ 1.   2.8  4.6]
#                         [ 6.4  8.2 10. ]]
g=np.linspace(1,10,6).reshape((2,3)) # 将1到10（包括10）分成6段，然后变成2行3列
print(g)
```



##### 2.numpy的计算01

```python

import numpy as np

a=np.array([10,20,30,40])
b=np.arange(4) # 0,1,2,3
print(a,b)

# 矩阵加法，对应位置元素相加
c=a+b # [10 21 32 43]
print(c) 

# 取幂，对应位置元素求平方
d=b**2
print(d)

# 对a中的所有元素求sin(),再乘10
e=10*np.sin(a) # 对a里的每一个值求sin，然后再乘以10
print(e)

print(b)
# 对b中元素，对应位置是否小于3输出对应位置输出true，false
print(b<3)

a=np.array([[1,1],
            [0,3]])
b=np.arange(4).reshape((2,2))

print(a)
print(b)

# a中元素和b中元素对应位置相乘
c=a*b # 对应位置相乘

# np.dot才是矩阵乘法
c_dot=np.dot(a,b) # 矩阵点乘（矩阵乘法）

print(c)

print(c_dot)

c_dot_2=a.dot(b) # 矩阵乘法
print(c_dot_2)

# 求最值

# 输出一个2行4列在0~1之间的随机数[[0.46777022 0.7863286  0.15399931 0.76660397]
#                           [0.60700812 0.53520991 0.41259487 0.20697263]]
a=np.random.random((2,4))
print(a)
print(np.sum(a))# 3.936487640650547

print(np.sum(a,axis=1))# 在每一行中求和，输出每一行的最大值[2.17470211 1.76178553]

print(np.max(a)) # 输出a中的最大值0.7863286024356131
print(np.max(a,axis=0))# 在每一列中求最大值，输出每一列的最大值[0.60700812 0.7863286  0.41259487 0.76660397]
print(np.min(a)) # 0.15399931422555357


```



##### 3.numpy的计算02

```python
import numpy as np

a=np.arange(2,14).reshape(3,4)
print(a) # [[ 2  3  4  5]
#           [ 6  7  8  9]
#           [10 11 12 13]]

# 输出矩阵a的转置
print(np.transpose(a)) # [ 2  6 10]
#                        [ 3  7 11]
#                        [ 4  8 12]
#                        [ 5  9 13]]
# 矩阵乘法，a的转置乘以a
print(np.transpose(a).dot(a)) # [[140 158 176 194]
#                                [158 179 200 221]
#                                [176 200 224 248]
#                                [194 221 248 275]]

# 输出矩阵a的均值7.5
print(np.mean(a))

# 输出矩阵a的中位数7.5
print(np.median(a))

# 逐位累加
print(np.cumsum(a)) # [ 2  5  9 14 20 27 35 44 54 65 77 90] 逐个相加

# 逐个相减
print(np.diff(a)) # [[1 1 1]
#                    [1 1 1]
#                    [1 1 1]]

print(np.nonzero(a)) # (array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int64), array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=int64)
# 输出对应的行和列是对应的不是0的位置，返回的是非0下标的索引，前一个是行，后一个是列

a=np.arange(0,14).reshape(2,7)
print(a)
print(np.nonzero(a)) # (array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=int64), array([1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6], dtype=int64))

a=np.arange(14,2,-1).reshape(3,4)
print(a)        # [[14 13 12 11]
#                  [10  9  8  7]
#                  [ 6  5  4  3]]
print(np.sort(a)) # 对每一行进行排序 [[11 12 13 14]
#                                  [ 7  8  9 10]
#                                  [ 3  4  5  6]]
print(a.T)
```





##### 4.索引

```python
# index
a=np.arange(3,15).reshape(3,4)
print(a) # [[ 3  4  5  6]
#           [ 7  8  9 10]
#           [11 12 13 14]]

# 下标从0开始,输出第二行
print(a[2]) # [11 12 13 14]

# 以下两种方式都可以
print(a[2][2]) # 13
print(a[2,2]) # 13
print(a[2,:]) # 输出这一整行 [11 12 13 14]
print(a[:,2]) # 输出这一整列  [ 5  9 13]

print(a[2,1:2]) # 输出第3行在2列的数 [12]
print(a[2,1:3]) # 输出第3行在第2列和第3列的数 [12 13]
for row in a:
    print(row) # 依次输出每一行

for col in a.T:
    print(col) # 这里先做了转置，依次输出每一列

# 输出每一个元素，可以先展平，在用for循环遍历
print(a.flatten()) # [ 3  4  5  6  7  8  9 10 11 12 13 14]

# 或者，构造a展平
for item in a.flat: 
    print(item)

```



##### 5.合并

```python
# 合并

b=np.array([1,1,1])
c=np.array([2,2,1])
d=np.vstack((b,c)) # 纵向合并
print(d)   # [[1 1 1]
#            [2 2 1]]

d=np.hstack((b,c)) # 横向合并
print(d) # [1 1 1 2 2 1]

# 将1行3列装换为3行1列
print(b.reshape(3,1))
# [[1]
#  [1]
#  [1]]
```



##### 6.拆分

```python
b=np.arange(12).reshape(3,4)
print(b) # [[ 0  1  2  3]
#           [ 4  5  6  7]
#           [ 8  9 10 11]]

# 可以不等拆分
print(np.array_split(b,3,axis=1))
# [array([[0, 1],
#         [4, 5],
#         [8, 9]]), array([[ 2],
#                          [ 6],
#                          [10]]), array([[ 3],
#                                         [ 7],
#                                         [11]])]
print(np.split(b,2,axis=1))
# [array([[0, 1],
#         [4, 5],
#         [8, 9]]), array([[ 2,  3],
#                          [ 6,  7],
#                          [10, 11]])]



print(np.hsplit(b,2)) # 横向分成2块
# [array([[0, 1],
#         [4, 5],
#         [8, 9]]), array([[ 2,  3],
#                          [ 6,  7],
#                          [10, 11]])]
print(np.vsplit(b,3)) # 纵向分成3块
# [array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]
```



##### 7.拷贝

```python
# 拷贝
# 浅拷贝是引用，深拷贝是新建

a=np.array([11,22,33,44])
b=a # 浅拷贝
c=a
d=b
a[0]=1
print(b) # [ 1 22 33 44]

a=np.array([100,255,265,322])
b=a.copy() # 深拷贝
a[0]=0
print(a) # [  0 255 265 322]
print(b) # [100 255 265 322]
```

