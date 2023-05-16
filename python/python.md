



### 1 python 安装

#### 1.1 python的安装

浏览器搜索python.org进入python官网下载即可

![1663380614297](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1663380614297.png)

**这里添加python路径一定要勾上**

![1663380826484](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1663380826484.png)

后续步骤默认点下一步就好了

#### 1.2 验证

win+R打开cmd窗口，输入python后看到版本号就表明安装成功

![1663381277265](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1663381277265.png)

输入python代码就可以运行了，一定要进入python里面，出现>>>符号后才可以运行python代码

![1663381430013](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1663381430013.png)

#### 1.3 python解释器

![1663381651776](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1663381651776.png)

运行解释器程序

新建一个文本文档，将后缀改为py就可以

用记事本打开，输入python代码，后保存

![1663382293753](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1663382293753.png)

然后打开cmd窗口，输入python，找到文件的路径即可回车运行

![1663382450607](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1663382450607.png)

#### 1.4 python开发环境

首先，我们先下载并安装它：

\- 打开网站：https://www.jetbrains.com/pycharm/download/#section=windows

下载PyCharm，专用于python的开发环境

安装好后，可以下一个这个插件，方便翻译

![1664010362855](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1664010362855.png)



### 2 入门语法

#### 2.1 标识符

```python
# 标识符
""""
规则1：内容限定，限定只能使用：中文，英文，数字，下划线，注意不能以数字开头
规则2：大小写敏感
规则3：不可以使用关键字


规范1:见名知意
规范2：多个单词组合时，使用下划线做分隔
命名中的英文字母，全部小写




单引号定义法：name='李春'
双引号定义法：name="李春"
三引号定义法：name="""  """

如果字符串本身是包含单或双引号的
解决:单引号定义法，可以内含双引号
    双引号定义法，可以内含单引号
    使用转义字符 \ 解除引号的作用
"""

# 字符串拼接，使用+ 就可以完成字符串或字符串字面量的拼接
# 通过占位的形式，完成拼接

# 字符串格式化
# %d，将内容转为整数，放入占位位置
# %f，将内容转为浮点数，放入占位位置
# 字符串占位符 %s
# % 表示我要，占位
# s 表示：将变量变成字符串放入占位的地方

# 精度控制 %m.nf
# 注意m比数字本身的宽度还小，不生效
# .n做小数精度的控制，会四舍五入，数据会失真


# 第二种字符串格式化的方式，快速格式化
# f"{占位}"  不会理会类型，不做精度控制

# 获取键盘的输入信息
# input语句不管你输入的是什么内容，都当字符串来看待，需要注意

```

#### 2.2 布尔类型和比较运算符

```python
bool_1=True
print(f"bool_1的内容是：{bool_1}，类型是：{type(bool_1)}")

# 比较运算符的使用
# == !=  >  <  >=  <=
num1=10
num2=15
print(f"num1是否大于num2：{num1>num2}")

# 条件判断语句
# 判断条件后的是冒号:
# 归属于if语句的代码块，需在前方填充4个空格,（回车就行）
age=17
if age>=18:
    print("我已经成年了")
    print("即将步入大学生活了")
print("hello world")

year=22
#year=int(input("请输入你的年龄："))   # 这里input是字符串类型，里面的内容是提示语，输入的内容需要强制类型转换一下
if year>=18:
    print("您已成年，您需要补票10元")
print("祝您游玩愉快")


# if else 语句
# age=int(input("请输入您的年龄："))
if age>=18:
    print("您已成年，需要补票")
else:
    print("您可以免费游玩")


# if elif else语句，条件判断多条件是互斥的
height=120
if height>=180:         # 也可以将input语句集成到条件判断语句当中
    print("需要全票")
elif height>=120:
    print("需要半票")
else:
    print("身高小于120，您可以免费")


# if条件语句的嵌套
age =int(input("请输入您的年龄："))
if age>=18&age<=30:
    print("您满足基本资格")
    if int(input("您的工作年限为："))>=2:
        print("满足基本工作年限条件，您可以领取")
    elif int(input("您的等级是："))>=3:
        print("您的等级满足，您可以领取")
    else:
        print("工作年限或等级不足，不可领取")
else:
    print("抱歉，您没有领取资格")
```



#### 2.3 for循环

```python
# for循环
# 语法：for零时变量in待处理的数据集（序列类型）
```

> 语法1：range(num)  获取一个从0开始，到num结束的数字序列（不含num本身）
> 语法2：range(num1，num2)，获取一个从num1开始，到num2结束的数字序列，但不含num2本身
> 语法3：range(num1，num2，step)，获取一个从num1开始，到num2结束的数字序列，步长为step，但不含num2本身
>
> for循环中
>
> 临时变量，在编程上，作用域范围上，只限定for循环内部使用
>
> 在其他编程语言，作用域范围内的变量范围外是访问不到的
>
> 实际上是可以访问的
>
> 规范上是不允许的，不建议使用
>
> 可以扩大作用域范围，如全局变量



```python
# for循环
# 语法：for零时变量in待处理的数据集（序列类型）

""""
语法1：range(num)  获取一个从0开始，到num结束的数字序列（不含num本身）
语法2：range(num1，num2)，获取一个从num1开始，到num2结束的数字序列，但不含num2本身
语法3：range(num1，num2，step)，获取一个从num1开始，到num2结束的数字序列，步长为step，但不含num2本身
"""

for x in range(10):
    print(x)

print("==================")
for x in range(5,10):
    print(x)

print("==================")
for x in range(5,10,2):
    print(x)

# x也可以不用
for x in range(10):
    print("李春")


# for循环中
# 临时变量，在编程上，作用域范围上，只限定for循环内部使用
# 在其他编程语言，作用域范围内的变量范围外是访问不到的
# 实际上是可以访问的
# 规范上是不允许的，不建议使用
# 可以扩大作用域范围，如全局变量
for i in range(5):
    print(i)
print(i)

# for循环嵌套
# 默认缩进是4个空格，回车即可
# 向郑金财表白100天
# 每天送10朵玫瑰花
i=1
for i in range(1,101):
    print(f"今天是喜欢郑金财的第{i}天，坚持")
    for j in range(1,11):
        print(f"送给郑金财的第{j}朵玫瑰花")
    print(f"我喜欢郑金财的第{i}天表白结束")

print(f"累计{i}天，表白成功")



# 循环中断break，continue
# continue 中断本次循环，进入下一次循环，作用范围，当前循环，对外侧循环无效
# break 直接结束当前循环



for i in range(1,6):
    print("语句1")
    for i in range(1,6):
        print("语句2")
        break
        print("语句3")
    print("语句4")
```

#### 2.4 函数

> 函数定义语法
> def 函数名(传入参数）:
>     函数体
>     return 返回值
>
> 参数和返回值都可以省略
> 不返回默认返回None
> 在if判断时，None等同于False
> 函数的返回值通过变量接收

``` python
# 统计字符串长度
str1 = "awevasdsd"
count = 0
for i in str1:
    count+=1
print(f"{str1}的长度为{count}")


# 定义函数
def my_length(str):
    """
    多行注释，回车，自动生成函数说明文档
    :param str: 参数str
    :return: 返回结果
    """
    count=0
    for i in str:
        count+=1
    print(f"{str}的长度为{count}")


my_length("dfhbsfbfgn")



""""
函数定义语法
def 函数名(传入参数）:
    函数体
    return 返回值


参数和返回值都可以省略
不返回默认返回None
在if判断时，None等同于False
函数的返回值通过变量接收
"""



def check1():
    print("你好，李春\n 请出示您的健康码")

check1()


def sum(num1,num2):
    print(f"{num1}+{num2}和为{num1+num2}")

sum(12,30)


def check():
    print("欢迎来到广州大学黄埔研究生院，请出示您的健康码，并配合检查体温")
    num1=int(input("请输入您的体温"))
    if num1 <=37.5:
        print(f"您的体温为{num1},体温正常，请进")
    else :
        print(f"您的体温为{num1}，需要隔离")


check()


# 函数说明文档
# 多行注释，回车，自动生成函数说明文档

def check2():
    """
    这是一个检查体温的函数
    :return:
    """
    print("欢迎来到广州大学黄埔研究生院，请出示您的健康码，并配合检查体温")
    num1=int(input("请输入您的体温"))
    if num1 <=37.5:
        print(f"您的体温为{num1},体温正常，请进")
    else :
        print(f"您的体温为{num1}，需要隔离")

# 鼠标悬停在函数名上，可以看到函数说明文档
check2()


# 函数内的局部变量和全局变量同名的时候，要想通过局部变量修改全局变量的值，在局部变量上面加global关键字即可

num4 = 100
def fun1():
    global num4
    num4 = 200
    print(num4)

print(num4)
fun1()
print(num4)

```

#### 2.5 数据容器

> 数据容器，一份变量，可以容纳多份数据

```python
# 数据容器，一份变量，可以容纳多份数据

# 元素类型不受限制
name_list = ['zjc','sdhgrds',True,False,'郑金财']


print(name_list)
# 输出指定下标的内容,下标从0开始

print(name_list[3])

name1_list=[[1,2,3],[4,5]]
print(name1_list[0][1])
print(name1_list[-1][-2])
"""
python下标索引还支持反向取数据
name[-1]是取出倒数第一个数据
name[-1][-2]是取出倒数第一个数据集的倒数第二个数据
"""

my_list=["tom","mary","djfhsadjasg"]
for i in range(len(my_list)):
    print(my_list[i])

```

#### 2.6 列表

```python
"""
列表是可以修改的
"""


mylist = ["ddfvsd","ugsdcas","kdfjhedsai","zjc","zhengjincai"]
# 查找某元素在列表的下标索引
index = mylist.index("zjc")
print(index)

# 查找某元素在列表不存在得到情况会报错ValueError
# 2.在指定的下标位置，插入指定的元素
mylist.insert(1,"best")
print(f"列表list插入元素后为{mylist}")

# 3.列表尾部追加单个元素   append方法
mylist.append("郑金财")
print(f"列表追加元素后为{mylist}")

# 4.列表追加一批元素   extend方法
mylist2=[1,2,6,8,9,True]
mylist.extend(mylist2)
print(f"列表追加一批元素后结果是{mylist}")


mylist = ["ddfvsd","ugsdcas","kdfjhedsai","zjc","zhengjincai"]
# 5.删除元素
# 5.1 删除指定下标元素 del
del mylist[2]
print(f"列表删除元素后为{mylist}")
# 5.2 删除指定下标元素 pop方法取出元素
mylist = ["ddfvsd","ugsdcas","kdfjhedsai","zjc","zhengjincai"]
element = mylist.pop(2)
print(f"列表取出的元素是{element}，此时列表为{mylist}")


# 5.3 删除某元素在列表的第一个匹配项  remove方法按元素内容清除匹配的第一个
mylist = ["ddfvsd","ugsdcas","kdfjhedsai","zjc","zjc","zhengjincai","zjc"]
mylist.remove("zjc")
print(f"通过remove方法后删除元素后，列表为{mylist}")

# 6 清空列表 clear
mylist.clear()
print(f"列表被清空了，结果是{mylist}")


# 7统计某元素在列表的数量 count方法,返回该元素的个数

mylist = ["ddfvsd","ugsdcas","kdfjhedsai","zjc","zhengjincai","zjc","zjcc"]
num1 = mylist.count("zjc")
print(f"zjc的个数为{num1}，此时列表为{mylist}")

# 8统计列表元素的个数
mylist = ["ddfvsd","ugsdcas","kdfjhedsai","zjc","zhengjincai","zjc","zjcc"]
print(f"列表为{mylist}，列表的元素个数是{len(mylist)}")


# 遍历列表
# while循环遍历列表

mylist = ["ddfvsd","ugsdcas","kdfjhedsai","zjc","zhengjincai","zjc","zjcc"]
index=0
while index<len(mylist):
    print(f"列表的第{index+1}个元素是{mylist[index]}")
    index+=1

print("while循环遍历完成")
print("====================")

# for循环遍历列表
"""
# 对于列表的每一个元素i，这里的i指的是元素，不是下标
"""

for i in mylist:
    print(f"列表的元素是{i}")
print("for循环遍历完成")

print("=====================")

# 用函数遍历
def bianli(mylist):
    for i in range(len(mylist)):
        print(f"列表的第{i + 1}个元素是{mylist[i]}")
    print("函数遍历完成")

bianli(mylist)


mylist=[1,2,3,4,5,6,7,8,9,10]
mylist01=[]
for element in mylist:
    if element%2==0:
        mylist01.append(element)
print(f"偶数列表为{mylist01}")

```

#### 2.7 元组

> tuple元组定义：定义元组使用小括号，且使用逗号隔开各个数据，数据可以是不同的数据类型
> 元组一旦定义不能修改
> del remove insert就不能使用了
>
> 元组虽然不能修改内容，但有一个特例，如果元组里面嵌套了一个列表，就可以修改列表里的内容，但是不可以修改元组的结构

```python
"""
tuple元组定义：定义元组使用小括号，且使用逗号隔开各个数据，数据可以是不同的数据类型
元组一旦定义不能修改
del remove insert就不能使用了
"""

t1 = (1,2,"zjc",True)
t2 = ()
t3 = tuple()
print(f"t1的类型是{type(t1)},内容是{t1}")

# 定义单个元素的元组必须在元素后面加逗号，否则就不是元组类型了

t4 = ("zjcc",)
print(f"t4的类型是{type(t4)},内容是{t4}")

# 查找元组元素
print(t1[2])

# 查找元组中某个元素的下标
index = t1.index("zjc")
print(index)


# 统计元组中某个元素的个数
print(t1.count("zjc"))
print(len(t1))


"""
元组虽然不能修改内容，但有一个特例，如果元组里面嵌套了一个列表，就可以修改列表里的内容，但是不可以修改元组的结构
"""

t5 = (1,2,3,["saedfga","sergter","zjc"])
t5[3][0]="zjc"
t5[3][1]="zjcc"
t5[3][2]="zjczjc"
print(t5)


t6 = ('周杰伦',11,['football',"music"])
print(f"年龄{t6[1]}所在元组的下标是{t6.index(11)}")
print(f"学生的姓名为：{t6[0]}")
del t6[2][0]

#这里t6[2]是列表
t6[2].append("coding")
print(t6)
```



#### 2.8 字符串

> 字符串也是不可修改的,不可以修改字符串的某一个字符，字符串替换是会得到一个新的字符串，原字符串并未改动

```python
"""字符串也是不可修改的"""


my_string = "zjczjcqqsddcsgfxc"
element1 = my_string[2]
element2 = my_string[12]
#取字符串倒数第一个元素
element3 = my_string[-1]

print(my_string)
#my_string[2] = "s"
# 字符串里的内容是不支持修改的

my_string = "zjcddcsgfxc"
print(my_string)
print(element1,element2,element3)


# 字符串替换，不会修改原字符串，但会返回一个新的字符串
new_my_string=my_string.replace("zj","qwqwqw")
print(new_my_string)


print(my_string.count("zj"))

print(len(my_string))


my_string="zjc zjc 郑金财 广州大学"
print(my_string.count("zjc"))
print(my_string.replace(" ","|"))
print(my_string.split("|")) # 返回一个列表
print(my_string.split(" ")) # 会取出字符串开头和结尾的空格以及换行符


"""
序列，切片的操作
切片操作并不会影响序列本身，而是会得到一个新的序列
语法：序列[起始下标；结束下标；步长]
结束下标不包括，从起始位置到末尾位置，可以正序可以逆序

"""


my_string = "万过薪月，员序程马黑"
print(my_string[-1:-6:-1])

set1 = {1,2,3}
set2 = {1,5,6}
print(set1.difference_update(set2))     # ????????
print(len(set1))
```

#### 2.9 集合

> 集合用大括号去定义，集合支持修改，但是不存在重复元素，自带去重效果
> 因为集合是无序的，所以集合不支持下标访问
> 如果定义空集合 set()即可
>
> 集合的遍历
>
> 不能用while循环，但是可以用while循环



```python
"""
集合用大括号去定义，集合支持修改，但是不存在重复元素，自带去重效果
因为集合是无序的，所以集合不支持下标访问
如果定义空集合 set()即可
"""

my_set = {"zxcz","dfas","sdav","zjc","zjc"}
print(my_set)

# 添加新元素
my_set.add("zjc")
my_set.add("郑金财")   # 添加的情况无序
print(my_set)

# 移除元素
my_set.remove("zjc")
print(my_set)

# 随机移除元素
elememt = my_set.pop()
print(elememt)
print(my_set)

set1 = {1,2,3}
set2 = {2,3,4}
# 求差集
print(set1.difference(set2))

set1 = {1,2,3}
set2 = {1,3,4}
# 消除差集
set3 = set2.difference_update(set1) #??????
print(set3)


print(set1.union(set2))


print(len(set1))


# 集合的遍历
# 不能用while循环，但是可以用while循环


set4 = {1,2,3,6,4,5,9,8}
for element in set4:
    print(element)


# 练习
my_list = [1,2,3,6,4,0,9,9,"zjc","zjc"]

# 定义空集合
set5 = set()
print(type(set5))
for element in my_list:
    set5.add(element)
print(set5)


print(type(set6))


```

#### 2.10 字典

> 字典，和集合一样，用大括号去定义，里面的元素是键值对的形式，通过对应的key，就可以找到对应的value
> 字典中的key也是不可以重复的
> 定义空字典 {}
> 或者my_dict = dict()

```python
"""
字典，和集合一样，用大括号去定义，里面的元素是键值对的形式，通过对应的key，就可以找到对应的value
字典中的key也是不可以重复的
定义空字典 {}
或者my_dict = dict()
"""

my_dict = {"李白":77,"宝玉":88,"黛玉":90,}

print(my_dict["李白"])
print(my_dict["黛玉"])

# 字典的key和value可以是任意数据类型，但key不可为字典

scores = {
    "李白":{"语文":123,"english":88,"数学":80},
    "韩信":{"语文":103,"english":98,"数学":120},
    "宝玉":{"语文":149,"english":68,"数学":100}
}

# 查看李白的english成绩
print(scores["李白"]["english"])


my_dict = {"李白":77,"宝玉":88,"黛玉":90,}

# 新增元素
my_dict["黛玉"]=100
print(f"新增后的字典为{my_dict}")

# 更新元素
my_dict["李白"]=66
print(f"更新后的字典为{my_dict}")

# 获取字典中全部的key
keys = my_dict.keys()
print(f"字典中的全部key是{keys}")

# 拿到key后可以通过key可以遍历字典
for key in keys:
    print(my_dict[key])


# 统计序列长度len()

# Max最大元素
my_tuple = (1,2,3,6,7)
print(max(my_tuple))

# Min最小元素
my_string = "wsedcasc"
print(max(my_string))


# 容器转列表
my_list = [1,2,3,4,5,6,7]
my_tuple = (1,4,5,9,88,45)
my_string = "sdvsdfvdf"
my_set={1,2,3,25,451}
my_dict={"key1":1,"key2":2,"key3":3}

print(list(my_list))
print(list(my_tuple))
print(list(my_string))# 将字符串转成列表，拆成单个字符
print(list(my_set))
print(list(my_dict))# 将字典的所有的keys转成列表


# 容器转元组
print(tuple(my_list))
print(tuple(my_tuple))
print(tuple(my_string))# 将字符串转成列表，拆成单个字符
print(tuple(my_set))
print(tuple(my_dict))# 将字典的所有的keys转成列表



# 容器转字符串
print(str(my_list))# 所有都是字符串，但显示是去掉引号的
print(str(my_tuple))
print(str(my_string))# 将字符串转成列表，拆成单个字符
print(str(my_set))
print(str(my_dict))# 将字典的所有的keys转成列表

# 容器转集合
print(set(my_list))# 所有都是字符串，但显示是去掉引号的
print(set(my_tuple))
print(set(my_string))# 将字符串转成列表，拆成单个字符
print(set(my_set))
print(set(my_dict))# 将字典的所有的keys转成列表，但顺序是混乱的


# sorted排序函数
print(sorted(my_list))
print(sorted(my_tuple))
print(sorted(my_string))
print(sorted(my_set))
print(sorted(my_dict)) # 对key值排序

# 降序sorted(序列,reverse=True)
print(sorted(my_list,reverse=True))
print(sorted(my_tuple,reverse=True))
print(sorted(my_string,reverse=True))
print(sorted(my_set,reverse=True))
print(sorted(my_dict,reverse=True)) # 对key值排序
```

#### 2.11 函数进阶

> 1.函数的多返回值，就是根据位置去接收对应返回值就可以了
> 2.位置参数，根据调用函数时根据函数定义的参数位置来传递参数
> 3.关键字参数：函数调用时，通过“键=值”形式传递参数
>     让函数更加清晰，容易使用，也清除了参数顺序需求
>     位置参数和关键字参数混用的时候，必须将位置参数放在最前面
> 4.缺省参数（默认值）默认值参数必须放在参数列表的后面
>
> 5.不定长参数，参数的个数是不确定的
> - 位置不定长，*号  一般命名为args
> 不定长定义的形式参数会作为元组存在，接收不定长数量的参数传入
> - 关键字不定长，**号，以字典的形式去接收kwargs
>
> 
>
> 6.函数作为参数传递
> 函数compute作为参数传入，这个函数需要接收两个数字进行计算，计算逻辑由这两个北传入函数决定
> 这只是一种计算逻辑的传入，而非数值的传递
>
> 就像上述代码一样，相加，相除都可以作为逻辑传入的,这里强调的逻辑是不固定的
>
> 7.匿名函数 lambda
> 语法lambda 传入参数：函数体（一行代码）
> 匿名函数用于临时构建一个函数，只用一次的场景

```python

def test01():
    return 1,2,True

x,y,z=test01()
print(x)
print(y)
print(z)

def test02(name,age,gender):
    print(f"您的名字是：{name}，年龄是：{age}，性别是：{gender}")

test02("李白",16,"男")

def test03(name,age,gender):
    print(f"您的名字是：{name}，年龄是：{age}，性别是：{gender}")

test03(name="李白",gender="男",age=18)


def test04(name,age,gender):
    print(f"您的名字是：{name}，年龄是：{age}，性别是：{gender}")

test04("李白",gender="男",age=22)


def test05(name,age=26,gender="女"):
    print(f"您的名字是：{name}，年龄是：{age}，性别是：{gender}")

test05("李白")
test05("李白",gender="男",age=22)


def user_info(*args):
    print(f"args参数类型是：{type(args)},内容是{args}")
user_info(1,2,3,True,"zjc")

def user_info1(**kwargs):
    print(f"args参数类型是：{type(kwargs)},内容是{kwargs}")
user_info1(name="李白",age=16,number=1090)


def test_func(compute):
    print(type(compute))
    result = compute(1,2)
    print(result)
def compute(x,y):
    return x + y
test_func(compute)


def test_func01(compute):

    result = compute(2,5)
    print(f"结果是{result}")
test_func01(lambda x,y:x*y)
```

#### 2.12 文件操作

> 使用with open可以自动的关闭文件语句块
> with open() as A
>
> 文件写入 w模式
>
> 文件不存在，则会创建
> 若文件存在，继续写入，会覆盖原本的内容
> open()
> write()写入内存中
> flush刷新，写到硬盘里
> close关闭
>
> 写文件，追加写文件 a模式
> a模式，文件不存在，会创建新文件
> 文件存在，则会在原有内容追加写入

```python
import time

f = open("D:/test.txt","r",encoding="UTF-8")
print(type(f))
# 读取10个字节的内容
#print(f.read(10))
#print(f.read())


#print(f.readlines())# 读取的全部行放到list里面
# readline()一次读取一行
for line in f:
    print(line)


# time.sleep(500000)# 演示文件占用
# 文件的关闭
f.close()

#time.sleep(500000)

with open("D:/test.txt","r",encoding="UTF-8") as f:
    for line in f:
        print(line)



# 文件写入
#f = open("D:/test01.txt","w",encoding="UTF-8")  # 如果文件名，不存在，文件会清空
#f.write("hello world!!!")
#time.sleep(600000)
#f.flush()
# time.sleep(600000)# 睡眠

f = open("D:/test01.txt","w",encoding="UTF-8")
f.write("郑金财")
#time.sleep(600000)
f.flush()
f.close()

# 追加写文件，a模式
f = open("D:/test01.txt","a",encoding="UTF-8")
f.write("\n郑金财是帅比")
f.close()

```

#### 2.13 异常捕获

```python
"""

异常捕获
try:
    可能发生的错误代码
except:
    如果出现异常执行的代码


捕获指定的异常
捕获变量未定义的异常
try:

except NameError as e:
except (NameError,ZeroDivisionError) as e:


捕获所有异常：
try:

except Exception as e:
    print("出现异常了")


假设真的没有出现异常所执行的代码
try:
    f = open("D:/abc.txt", "r", encoding="UTF-8")
except Exception as e:
    print("出现异常了")
else:
    print("好高兴，没有出现异常")





try:
    f = open("D:/abc.txt", "r", encoding="UTF-8")
except Exception as e:
    print("出现异常了")
else:
    print("好高兴，没有出现异常")
finally:
    print("我是finally，有没有异常我都会执行“)
    f.close
"""


#   f = open("D:/abc.txt","r",encoding="UTF-8") #No such file or directory: 'D:/abc.txt'

try:
    f = open("D:/abc.txt", "r", encoding="UTF-8")
except:
    print("该文件不存在")
    f = open("D:/abc.txt", "w", encoding="UTF-8")


try:
    f = open("D:/1.txt", "r", encoding="UTF-8")
except Exception as e:
    print("出现异常了,文件不存在")
    f = open("D:/1.txt", "w", encoding="UTF-8")
else:
    print("好高兴，没有出现异常")
finally:
    print("我是finally，有没有异常我都会执行")
    f.close



# 异常的传递，有异常，不一定要在最底层try
def func1():
    print("func1开始执行")
    num=1/0
    print("func1结束执行")
def func2():
    print("func2开始执行")
    func1()
    print("func2结束执行")
def main():
    try:
        func2()
    except Exception as e:
        print(f"出现异常了，异常信息是{e}")
main()

```

### 3.python模块

#### 3.1 模块的导入

![1665037394158](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665037394158.png)

注意事项：

- from可以省略，直接import即可
- as别名可以省略
- 通过”.”来确定层级关系
- 模块的导入一般写在代码文件的开头位置
- from...import 是导入模块的具体方法，调用时直接使用函数名即可
- import导入模块时，调用方法需要，模块名.方法名使用

- import后面内容是啥，就要以那个内容做起始调用

#### 3.2 制作自定义模块

案例：新建一个Python文件，命名为my_module1.py，并定义test函数

![1665038844472](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665038844472.png)



注意: 
  每个Python文件都可以作为一个模块，模块的名字就是文件的名字. 也就是说自定义模块名必须要符合标识符命名规则

**测试模块**

在实际开发中，当一个开发人员编写完一个模块后，为了让模块能够在项目中达到想要的效果，
这个开发人员会自行在py文件中添加一些测试信息，例如，在my_module1.py文件中添加测试代码test(1,1)

![1665038577033](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665038577033.png)

问题: 
此时，无论是当前文件，还是其他已经导入了该模块的文件，在运行的时候都会自动执行`test`函数的调用
解决方案：

![1665038639605](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665038639605.png)

**注意事项**



![1665038673176](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665038673176.png)

注意事项：当导入多个模块的时候，且模块内有同名功能. 当调用这个同名功能的时候，调用到的是后面导入的模块的功能

![1665038747218](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665038747218.png)

如果一个模块文件中有`__all__`变量，当使用`from xxx import *`导入时，只能导入这个列表中的元素,可以控制import *的行为

![1665038798998](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665038798998.png)

**总结**

> 1. 如何自定义模块并导入？
> 在Python代码文件中正常写代码即可，通过import、from关键字和导入Python内置模块一样导入即可使用。
> 2. __main__变量的功能是？
> if __main__ == “__main__”表示，只有当程序是直接执行的才会进入if内部，如果是被导入的，则if无法进入
> 3. 注意事项
> 不同模块，同名的功能，如果都被导入，那么后导入的会覆盖先导入的
> __all__变量可以控制import *的时候哪些功能可以被导入

### 4 python包

#### 4.1 自定义包

> 从物理上看，包就是一个文件夹，在该文件夹下包含了一个 __init__.py 文件，该文件夹可用于包含多个模块文件
> 从逻辑上看，包的本质依然是模块

![1665039797405](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665039797405.png)

> 包的作用: 
>      当我们的模块文件越来越多时,包可以帮助我们管理这些模块, 包的作用就是包含多个模块，但包的本质依然是模块





> 步骤如下:
> ① 新建包`my_package`
> ② 新建包内模块：`my_module1` 和 `my_module2`
> ③ 模块内代码如下

![1665039877808](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665039877808.png)

> Pycharm中的基本步骤:
>
> [New]      [Python Package]    输入包名    [OK]     新建功能模块(有联系的模块)
>
> 注意：新建包后，包内部会自动创建`__init__.py`文件，这个文件控制着包的导入行为



**导入包：**

![1665039964606](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665039964606.png)

![1665040031140](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665040031140.png)



**总结：**

> 1. 什么是Python的包？
> 包就是一个文件夹，里面可以存放许多Python的模块（代码文件），通过包，在逻辑上将一批模块归为一类，方便使用。
> 2. __init__.py文件的作用？
> 创建包会默认自动创建的文件，通过这个文件来表示一个文件夹是Python的包，而非普通的文件夹。
> 3. __all__变量的作用？
> 同模块中学习到的是一个作用，控制 import * 能够导入的内容

#### 4.2安装第三方包

什么是第三方包？

> 我们知道，包可以包含一堆的Python模块，而每个模块又内含许多的功能。
> 所以，我们可以认为：一个包，就是一堆同类型功能的集合体。
>
> 在Python程序的生态中，有许多非常多的第三方包（非Python官方），可以极大的帮助我们提高开发效率，如：
> 科学计算中常用的：numpy包
> 数据分析中常用的：pandas包
> 大数据计算中常用的：pyspark、apache-flink包
> 图形可视化常用的：matplotlib、pyecharts
> 人工智能常用的：tensorflow
> 等

**安装第三方包 - pip**

> 第三方包的安装非常简单，我们只需要使用Python内置的pip程序即可。
>
> 打开我们许久未见的：命令提示符程序，在里面输入：
> pip install 包名称
> 即可通过网络快速安装第三方包

![1665046834526](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665046834526.png)

**pip的网络优化**

> 由于pip是连接的国外的网站进行包的下载，所以有的时候会速度很慢。
>
> 我们可以通过如下命令，让其连接国内的网站进行包的安装：
> pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 包名称

![1665046884408](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665046884408.png)

https://pypi.tuna.tsinghua.edu.cn/simple 是清华大学提供的一个网站，可供pip程序下载第三方包

**安装第三方包 - PyCharm**

![1665046952059](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665046952059.png)

觉得慢的话，在可选网址里加上-i https://pypi.tuna.tsinghua.edu.cn/simple

### 5 数据可视化

#### 5.1 JSON数据格式

什么是json？

- JSON是一种轻量级的数据交互格式。可以按照JSON指定的格式去组织和封装数据
- JSON本质上是一个带有特定格式的字符串
- 主要功能：json就是一种在各个编程语言中流通的数据格式，负责不同编程语言中的数据传递和交互. 类似于：
  - 国际通用语言-英语
  - 中国56个民族不同地区的通用语言-普通话

**json有什么用？**

> 为了让不同的语言都能够相互通用的互相传递数据，JSON就是一种非常良好的中转数据格式。如下图，以Python和C语言互传数据为例：

![1665051870348](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665051870348.png)

 **json格式数据转化**

![1665051923840](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665051923840.png)



**Python数据和Json数据的相互转化**

![1665051969370](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665051969370.png)



#### 5.2 pyecharts模块

- 如果想要做出数据可视化效果图, 可以借助pyecharts模块来完成

  概况 :

  > Echarts 是个由百度开源的数据可视化，凭借着良好的交互性，精巧的图表设计，得到了众多开发者的认可. 而 Python 是门富有表达力的语言，很适合用于数据处理. 当数据分析遇上数据可视化时pyecharts 诞
  >
  > 生了.



打开官方画廊：
https://gallery.pyecharts.org/#/README



**pyecharts模块安装**

> 使用在前面学过的pip命令即可快速安装PyEcharts模块，安装时间要等一小会
>
> pip install pyecharts

![1665046983204](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665046983204.png)



#### 5.3 基础折线图

> 基本步骤
>
> 读取数据
>
> 处理数据
>
> 装图
>
> 绘图

```python
from pyecharts.charts import Line # 导包
from pyecharts.options import TitleOpts, LegendOpts,ToolboxOpts,VisualMapOpts

# 创建一个折线图对象
line = Line()
# 添加x轴和y轴数据
line.add_xaxis(["中国","美国","日本"])
line.add_yaxis("GDP",[30,20,10])

# 设置全局配置选项
line.set_global_opts(
    title_opts=TitleOpts(title="GDP数据",pos_left="center",pos_bottom="1%"),
    legend_opts=LegendOpts(is_show=True),
    toolbox_opts=ToolboxOpts(is_show=True),
    visualmap_opts=VisualMapOpts(is_show=True)

)
line.render()   # 这里可以命名生成的文件名
# 此时会生成一个html文件
```

![1665452617980](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665452617980.png)

**全局配置选项**

这里全局配置选项可以通过set_global_opts方法来进行配置, 相应的选项和选项的功能如下:

![1665452670993](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665452670993.png)

![1665452706275](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665452706275.png)



#### 5.4 地图

```python
from pyecharts.charts import Map
from pyecharts.options import VisualMapOpts
map=Map()
# 这里的数据元组形式
data = [
    ("北京",99),
    ("江西",999),
    ("广东",699),
    ("台湾",199),
    ("上海",899),
]

map.add("测试地图",data,"china")# 这里不写默认是中国地图
# 设置全局选项
map.set_global_opts(
    visualmap_opts=VisualMapOpts(
        is_show=True,
        is_piecewise=True,
        pieces=[{"min":1,"max":9,"label":"1-9","color":"#CCFFFF"},
                {"min":10,"max":99,"label":"10-99","color":"#FF6666"},
                {"min":100,"max":999,"label":"100-999","color":"#990033"},
                ]
    )
)
# 绘图
map.render()
```

效果

![1665452959857](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665452959857.png)

**视觉映射器**

![1665453036327](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665453036327.png)



#### 5.5 时间柱状图

```python
from pyecharts.charts import Bar
bar = Bar()
# 添加x轴数据
bar.add_xaxis(["中国","美国","日本"])
# 添加y轴数据
bar.add_yaxis("GDP",[30,20,10])
# 绘图
bar.render()
```

![1665453451193](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665453451193.png)

反转x轴和y轴

> bar.reversal_axis() 

![1665453629277](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665453629277.png)

数值标签在右侧

> bar.add_yaxis("GDP",[30,20,10],label_opts=LabelOpts(position="right"))

![1665453789870](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665453789870.png)





**创建时间线**

- <font color="red">Timeline()-时间线</font>

  柱状图描述的是分类数据，回答的是每一个分类中『有多少？』这个问题. 这是柱状图的主要特点,同时柱状图很难动态的描述一个趋势性的数据. 这里pyecharts为我们提供了一种解决方案-<font color="red">时间线</font>

<font color="red">如果说一个Bar、Line对象是一张图表的话，时间线就是创建一个一维的x轴，轴上每一个点就是一个图表对象</font>

![1665454100894](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665454100894.png)



1. 创建时间线

![1665454180909](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665454180909.png)

2. 自动播放

![1665454209593](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665454209593.png)

3. 时间线设置主题

   直接在时间对象里添加属性

![1665454245282](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665454245282.png)



#### 5.6 列表的sort方法

> 在前面我们学习过sorted函数，可以对数据容器进行排序。
> 在后面的数据处理中，我们需要对列表进行排序，并指定排序规则，sorted函数就无法完成了。
> 我们补充学习列表的sort方法。
> 使用方式

列表.sort(key=选择排序依据的函数, reverse=True|False)

- 参数key，是要求传入一个函数，表示将列表的每一个元素都传入函数中，<font color="red">返回排序的依据，可以实现按列表的哪一列内容进行排序</font>
- 参数reverse，是否反转排序结果，True表示降序，False表示升序



实现

![1665454479321](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1665454479321.png)





### 6 类和对象

#### 6.1 初识对象

举个例子：类就相当于要打印的学生信息表格的模板，对象就是一份要填写的具体的表格，不同的对象即每一个学生的信息都不一样

类包括成员变量和成员方法，对应于类的属性和行为

```python
class Student:
    name=None

    # 只要成员方法要想访问成变量，必须通过self关键字
    def say_Hi(self):
        print(f"你好，我是：{self.name}")
    # 成员方法访问外部参数时，不需要通过self关键字
    def say_Hi2(self,msg):
        print(f"你好，我是：{self.name},{msg}")

# 调用该成员方法的时候，可以当self关键字不存在
stu1=Student()
stu1.name="张三"
stu1.say_Hi()

stu2=Student()
stu2.name="凌俊杰"

stu2.say_Hi2("不错呦，我看好你哦")

class Clock:
    id=None
    name=None

    def ring(self):
        import winsound
        winsound.Beep(2000,3000)

clock1=Clock()
clock1.name="起床铃"
clock1.id="00101001"
print(f"闹钟{clock1.name}响了，快起床！")
# 这个真的能响，建议减小声音
clock1.ring()
print(f"闹钟{clock1.name}已停止响铃！")

```

#### 6.2 构造方法

\_\_init\_\_( ) 可以实现为成员变量初始化赋值，对象创建时可直接传参

```python
class Student:
    name=None
    age=None
    telphone=None

    # 构造方法是__init__(),未创建对象都会执行的，可以通过构造方法给成员变量赋值，实现对象创建时同时传入参数
    # 构造类对象
    def __init__(self,name,age,telphone):
        self.name=name
        self.age=age
        self.telphone=telphone
        print("构造在创建对象前就会执行")

stu1=Student("邹洁伦",22,"178787878789")
print(stu1.name)
print(stu1.age)
print(stu1.telphone)

# 这里有个小疑惑，为什么创建第二个对象或这个构造方法还会执行一次？？？？
stu2=Student(name="林军节",telphone="178787878789",age=22)
print(stu2.name)
print(stu2.age)
print(stu2.telphone)
```



#### 6.3 魔术方法

\_\_init\_\_( ) 

\_\_str\_\_( ) 	以字符串的形式输出对象的内容，不重写的话，就是输出该对象的地址，意义不大

\_\_lt\_\_( ) 	可以判断两个对象小于的情况，可以比较对象的某个属性

\_\_le\_\_( ) 	可以判断两个对象小于等于的情况，可以比较对象的某个属性



#### 6.4 封装

> <font color="red">私有的成员变量和成员方法只有类内部的方法可以使用，类对象即外部对象不可以直接调用</font>
>
> 定义私有成员变量和私有成员方法均以__开头
>
> 虽然私有成员变量，类对象不能直接修改，但是可以通过调用类里面的方法，达到修改的目的

```python
class phone:
    __battery=2
    def __sole_chip(self):
        print("CPU以单核运行")

    def is5g_communication(self):
        if self.__battery>=1:
            print("5g通话进行中")
        else:
            self.__sole_chip()
            print("电量不足，暂不能提供5g通信，已切换为cpu单核运行模式")

phone=phone()
phone.is5g_communication()


print("======================")

class cellphone:
    __is_5g_enable=False

    def __check_5g(self):
        if self.__is_5g_enable:
            print("5g通话进行中")
        else:
            print("5g中断，切换为4g网络")

    def call_by5g(self):
        self.__check_5g()
        print("正在通话中")
    def open_5g(self):
        self.__is_5g_enable=True

cellphone=cellphone()
cellphone.call_by5g()
cellphone.open_5g()
cellphone.call_by5g()
```

#### 6.5 继承

单继承语法： class 子类 (父类名)：

多继承语法： class 子类 (父类名1，父类名2，，，，)

> 如果子类中确实或者没有什么需要更新的成员，<font color="red">可以使用pass关键字写到函数体中，避免语法错误</font>
>
> 当不同的父类名中有相同的成员属性时，优先使用继承的父类列表中最左边的父类的属性

```python
class phone:
    fingerprint=None
    device_id="00121200"
    def is_4g(self):
        print("使用4g通话")

class Nfc:
    device_id="10025612"
    def nfc(self):
        print("使用nfc功能")

class cellphone(phone,Nfc):
    pass

cellphone=cellphone()
cellphone.nfc()
cellphone.is_4g()

# 当不同的父类名中有相同的成员属性时，优先使用最左边的父类的属性
print(cellphone.device_id) # 00121200
```



**复写**

子类继承父类的成员属性和方法后，如果对其“不满意”，那么可以进行复写

即：在子类中重新定义同名的属性或方法即可



**调用父类同名成员：**

一旦复写父类成员，那么类对象调用成员的时候，就会调用复写后的新成员

如果需要使用被复写的父类的成员，需要特殊的调用方式：

方式1：

 - 调用父类成员
   	- 使用成员变量：父类名.成员变量
      	- 使用成员方法：父类名.成员方法(self)

方式2：

- 使用super()调用父类成员
  - 使用成员变量：super().成员变量
  - 使用成员方法：super().成员方法()

![1666269781732](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1666269781732.png)

**注意**：

​	只可以在子类内部调用父类的同名成员，子类的实体类对象调用默认是调用子类复写的

#### 6.6 类型注解

为什么需要类型注解：

主要功能：

	- 帮助第三方IDE工具（如PyCharm）对代码进行类型判断，协助代码做提示
	- 帮助开发者对变量进行类型注释



支持：

- 变量的类型注解
- 函数（方法）形参列表和返回值的类型注解



类型注解的语法

![1666270445780](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1666270445780.png)

![1666270512204](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1666270512204.png)



![1666270700772](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1666270700772.png)



**类型注解的局限：**

![1666270641291](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1666270641291.png)



**函数方法的类型注解-形参注解**

![1666270833261](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1666270833261.png)

![1666270885796](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1666270885796.png)



![1666270946748](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1666270946748.png)



**Union类型**

Union联合类型注解，在变量注解，函数（方法）形参和返回值注解中，均可使用

使用Union类型，必须先导包

> from typing import Union

![1666271107516](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1666271107516.png)

![1666271216054](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1666271216054.png)

#### 6.7 多态

> 实现同样的行为，传入不同的对象，就是不同的状态

实现多态步骤：

​	创建一个父类

​	创建两个子类，子类复写父类的方法

​	创建一个函数，参数类型是传入父类类型，函数体调用父类方法

​	分别创建对应的子类对象

​	将子类对象传入函数中，就可以实现同样的行为，传入不一样的对象，表现不一样的行为

```python
class AC:
    def cool(self):
        pass
    def hot(self):
        pass


class MD_ac(AC):
    def cool(self):
        print("美的空调制冷")
    def hot(self):
        print("美的空调制热")

class GL_ac(AC):
    def cool(self):
        print("格力空调制冷")
    def hot(self):
        print("格力空调制热")

def make_cool(ac:AC):
    ac.cool()

# 这里创建一个MD对象
md=MD_ac()
# 这里创建一个GL对象
gl=GL_ac()

"""
多态
"""
# 实现同样的行为，传入不同的对象，就是不同的状态
make_cool(md)	# 美的空调制冷
make_cool(gl)	# 格力空调制冷

```

### 7.1 pycharm中安装pytorch

第一步电脑win+R键打开命令行窗口，然后输入nvidia-smi，检查CUDA Version这个版本号

```bash
nvidia-smi
```

会看到如下页面显示

![1667032621734](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1667032621734.png)



 注意看这里版本是CUDA Version: 11.6

然后在浏览器窗口打开pytorch官网，进入首页往下滑，找到如下这个界面：

​	这里要注意一下，因为使用pycharm安装pytorch，选择pip和对应的显卡版本号就行

​	如果未找到，就点击左下角查看对应的之前的版本

![1667032827760](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1667032827760.png)



将下面这条指令复制，打开pycharm

![1667033134206](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1667033134206.png)



打开pycharm，创建一个新项目或者在已有的项目中界面中，在页面底部找到terminal打开

![1667033340381](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1667033340381.png)





将之前复制得到那条命令粘贴到这里，然后回车运行，等待下载完成即可

![1667033437845](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1667033437845.png)



最后输入如下代码检验pytorch是否安装成功

```python
import torch
print(torch.cuda.is_available())

结果为True
```



![1667033691143](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1667033691143.png)

```bash

```

