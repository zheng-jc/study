##### 1.基本图

matplotlib画图基于numpy



```python
import matplotlib.pyplot as plt

import numpy as np

x=np.linspace(-1,1,50)
# y=2*x+1
# y=x**2
y=x**3
plt.plot(x,y)
plt.show()
```

![1678882358688](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1678882358688.png)

##### 2.图例

图例可以同时显示多张图

```python
# 大窗口

import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(-3,3,50)
y1=2*x+1
y2=x**2

# figure开始的标志
plt.figure()
plt.plot(x,y1)

plt.figure()

plt.plot(x,y1)
plt.plot(x,y2,color="red",linewidth=1.0,linestyle='--')

plt.show()

```

![1678882480970](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1678882480970.png)



##### 3.轴的修改(描述)

```python
import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(-3,3,50)
y1=2*x+1
y2=x**2

# figure开始的标志

plt.figure()

plt.plot(x,y1)
plt.plot(x,y2,color="red",linewidth=1.0,linestyle='--')

# 设置x轴的取值范围
plt.xlim((-1,2))
plt.ylim((-2,3))

# 修改x轴y轴的描述
plt.xlabel('I am x')
plt.ylabel('I am y')

# 修改x轴y轴的标尺
new_ticks=np.linspace(-1,2,5)
print(new_ticks)
plt.xticks(new_ticks)

# plt.yticks([-2,-1,0,2,3,],
#             ["really bad","bad","normal","good","really good"])

plt.yticks([-2,-1,0,2,3,],
            [r"$really\ bad\alpha$",r"$bad$",r"$normal$",r"$good$",r"$really\ good$"])

plt.show()

```

![1678882825638](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1678882825638.png)

##### 4.轴的位置

```python
import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(-3,3,50)
y1=2*x+1
y2=x**2

# figure开始的标志

plt.figure()

plt.plot(x,y1)
plt.plot(x,y2,color="red",linewidth=1.0,linestyle='--')

# 设置x轴的取值范围
plt.xlim((-1,2))
plt.ylim((-2,3))

# 修改x轴y轴的描述
plt.xlabel('I am x')
plt.ylabel('I am y')

# 修改x轴y轴的标尺
new_ticks=np.linspace(-1,2,5)
print(new_ticks)
plt.xticks(new_ticks)

# plt.yticks([-2,-1,0,2,3,],
#             ["really bad","bad","normal","good","really good"])

plt.yticks([-2,-1,0,2,3,],
            [r"$really\ bad\alpha$",r"$bad$",r"$normal$",r"$good$",r"$really\ good$"])

# gca = "get current axis"
ax=plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# 设置x轴y轴是哪个轴
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# 设置x轴位于y为0的位置
ax.spines['bottom'].set_position(('data',0))
# 设置y轴位于x为0的位置
ax.spines['left'].set_position(('data',0))

plt.show()

```

![1678883114894](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1678883114894.png)

##### 5.legend指示栏

```python
import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(-3,3,50)
y1=2*x+1
y2=x**2

# figure开始的标志

plt.figure()

# 这里一定要加逗号，才能传到handles里面
l1,=plt.plot(x,y1,label='up')
l2,=plt.plot(x,y2,color="red",linewidth=1.0,linestyle='--',label='down')

# 设置x轴的取值范围
plt.xlim((-1,2))
plt.ylim((-2,3))

# 修改x轴y轴的描述
plt.xlabel('I am x')
plt.ylabel('I am y')

# 修改x轴y轴的标尺
new_ticks=np.linspace(-1,2,5)
print(new_ticks)
plt.xticks(new_ticks)

# plt.yticks([-2,-1,0,2,3,],
#             ["really bad","bad","normal","good","really good"])

plt.yticks([-2,-1,0,2,3,],
            [r"$really\ bad\alpha$",r"$bad$",r"$normal$",r"$good$",r"$really\ good$"])

# 设置图例
# plt.legend()
plt.legend(handles=[l1,l2],labels=['aa','zz'],loc='best')

plt.show()

```

![1678886864351](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1678886864351.png)

##### 6.批注

```python

import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(-3,3,50)
y1=2*x+1
y2=x**2

# figure开始的标志

plt.figure()

plt.plot(x,y1)
plt.plot(x,y2,color="red",linewidth=1.0,linestyle='--')

# 设置x轴的取值范围
# plt.xlim((-1,2))
# plt.ylim((-2,3))

# 修改x轴y轴的描述
plt.xlabel('I am x')
plt.ylabel('I am y')

# 修改x轴y轴的标尺
# new_ticks=np.linspace(-1,2,5)
# print(new_ticks)
# plt.xticks(new_ticks)


# gca = "get current axis"
ax=plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# 设置x轴y轴是哪个轴
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# 设置x轴位于y为0的位置
ax.spines['bottom'].set_position(('data',0))
# 设置y轴位于x为0的位置
ax.spines['left'].set_position(('data',0))

x0=1
y0=2*x0+1

# 用散点图的形式展现，这里只展示一个点
plt.scatter(x0,y0,s=50,color='blue')

plt.plot([x0,x0],[y0,0],'k--',lw=2.5)

# 方法1
plt.annotate(r'$2x+1=%s$'%y0,xy=(x0,y0),xycoords='data',xytext=(+30,-30),textcoords='offset points',
             fontsize=16,arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))


# 方法2
# 前两个参数设置内容的起始位的x，y坐标，后面放展示内容
plt.text(-3,5,r'$this\ is\ the\ some\ text\ \mu\ \sigma_i\ \alpha_t$')

plt.show()

```





![1678886944782](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1678886944782.png)

##### 7.条形图

```python
import matplotlib.pyplot as plt
import numpy as np

n=12
x=np.arange(n)
y1=(1-x/float(n))*np.random.uniform(0.5,1.0,n)
y2=(1-x/float(n))*np.random.uniform(0.5,1.0,n)

plt.bar(x,y1,facecolor='#9999ff',edgecolor='white')
plt.bar(x,-y2,facecolor='#9999ff',edgecolor='white')


# for x,y in zip(x,y1):
#     plt.text(x+0.04,y+0.05,'%.2f'%y,ha='center',va='bottom')
#
# for x,y in zip(x,y2):
#     plt.text(x+0.04,y-0.05,'%.2f'%y,ha='center',va='top')

for x,y,z in zip(x,y1,y2):
    plt.text(x+0.04,y+0.05,'%.2f'%y,ha='center',va='bottom')
    plt.text(x+0.04,-z-0.05,'%.2f'%z,ha='center',va='top')


plt.xlim(-.5,n)
plt.ylim(-1.25,1.25)

plt.show()
```



![1678887384830](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1678887384830.png)

##### 8.散点图

```python
import matplotlib.pyplot as plt
import numpy as np

n=1024
x=np.random.normal(0,1,n)
y=np.random.normal(0,1,n)

c=np.arctan2(y,x) # color value

plt.scatter(x,y,c=c,alpha=0.5)
plt.xlim((-1.5,1.5))
plt.ylim((-1.5,1.5))

# 传入空值
plt.xticks(())
plt.yticks(())

# plt.scatter(np.arange(5),np.arange(5))

plt.show()
```

![1678931507015](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1678931507015.png)

##### 9.等高线图

```python
import matplotlib.pyplot as plt
import numpy as np


def f(x,y):
    # the height function
    return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)

n=256
x=np.linspace(-3,3,n)
y=np.linspace(-3,3,n)

# 把x，y绑定成网格的输入值
X,Y=np.meshgrid(x,y)

# 划分区域的
plt.contourf(X,Y,f(X,Y),8,alpha=0.75,cmap=plt.cm.hot)

# 等高线,8的意思是9条线，分10个区域
C=plt.contour(X,Y,f(X,Y),8,colors='black',linewidths=.5)

# 给等高线加标签的,标签画在线里面
plt.clabel(C,inline=True,fontsize=10)

plt.xticks(())
plt.yticks(())
plt.show()
```

![1678932738290](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1678932738290.png)

##### 10.image图片

```python
import matplotlib.pyplot as plt
import numpy as np

a=np.random.random((28,28))

print(a)
plt.imshow(a,interpolation='nearest',cmap='bone',origin='upper')
plt.colorbar()

plt.xticks()
plt.yticks()
plt.show()
```

![1678934027299](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1678934027299.png)

##### 11.3D数据图

```python
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D



fig=plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

# 在3.4的版本已被弃用
# ax=Axes3D(fig)

X=np.arange(-4,4,0.25)
Y=np.arange(-4,4,0.25)

X,Y=np.meshgrid(X,Y)

R=np.sqrt(X**2+Y**2)

Z=np.sin(R)

ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))
# offset表示要投影在哪条线上，这里投影到z为-2的位置
ax.contourf(X,Y,Z,zdir='z',offset=-2,cmap='rainbow')
ax.set_zlim(-2,2)
plt.show()
```

![1678935637527](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1678935637527.png)

##### 12.画子图subplot

```python
import matplotlib.pyplot as plt

plt.figure()
# 画子图，画2行2列4个子图，这是第1个
plt.subplot(2,2,1)
plt.plot([0,1],[0,1])
plt.xticks(())

plt.subplot(2,2,2)
plt.plot([0,1],[0,2])

plt.subplot(2,2,3)
plt.plot([0,1],[0,3])

plt.subplot(2,2,4)
plt.plot([0,1],[0,4])

plt.figure()
# 画子图，画两行，第1行1列，第2行3列，最后一个参数表示第几个图
plt.subplot(2,1,1)
plt.plot([0,1],[0,1])
plt.xticks(())

# 第一行一个图占了3个子图的位置，所以第2行的第1个，实际是第4个图子图
plt.subplot(2,3,4)
plt.plot([0,1],[0,2])

plt.subplot(2,3,5)
plt.plot([0,1],[0,3])

plt.subplot(2,3,6)
plt.plot([0,1],[0,4])
plt.show()
```

![1678940003131](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1678940003131.png)



##### 13.画子图subplot2grid

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.figure()
# 第一个参数代表总体布局3*3，第一个图在第一行第一列，行跨1行，列3列
ax1=plt.subplot2grid((3,3),(0,0),colspan=3,rowspan=1)
ax1.plot([1,2],[1,2])
ax1.set_title('ax1_title')
ax2=plt.subplot2grid((3,3),(1,0),colspan=2)
ax3=plt.subplot2grid((3,3),(1,2),rowspan=2)
ax4=plt.subplot2grid((3,3),(2,0))
ax5=plt.subplot2grid((3,3),(2,1))

plt.show()


```



![1678952256013](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1678952256013.png)



```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.figure()

# 规模3*3
gs=gridspec.GridSpec(3,3)
# 第一个图，占第一行，列占满
ax1=plt.subplot(gs[0,:])

# 第二个图，占第2行，占前两列
ax2=plt.subplot(gs[1,:2])
# 第三个图，占第三列，占后两行
ax3=plt.subplot(gs[1:,2])

# 占倒数第一行，第一列
ax4=plt.subplot(gs[-1,0])
# 占倒数第一行，第二列
ax5=plt.subplot(gs[-1,-2])
# ax5=plt.subplot(gs[2,1])

plt.show()


```

![1678954171327](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1678954171327.png)





```python
import matplotlib.pyplot as plt
# 画一个3行3列的，共享x轴y轴
f,((ax11,ax12,ax13),(ax21,ax22,ax23),(ax31,ax32,ax33))=plt.subplots(3,3,sharex=True,sharey=True)
ax11.plot([1,2],[1,2])

# plt.tight_layout()
plt.show()

```

![1678955310690](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1678955310690.png)

##### 14.图中图

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# plt.figure()

# # 规模3*3
# gs=gridspec.GridSpec(3,3)
# # 第一个图，占第一行，列占满
# ax1=plt.subplot(gs[0,:])
#
# # 第二个图，占第2行，占前两列
# ax2=plt.subplot(gs[1,:2])
# # 第三个图，占第三列，占后两行
# ax3=plt.subplot(gs[1:,2])
#
# # 占倒数第一行，第一列
# ax4=plt.subplot(gs[-1,0])
# # 占倒数第一行，第二列
# ax5=plt.subplot(gs[-1,-2])
# # ax5=plt.subplot(gs[2,1])

# 画一个3行3列的，共享x轴y轴
# f,((ax11,ax12,ax13),(ax21,ax22,ax23),(ax31,ax32,ax33))=plt.subplots(3,3,sharex=True,sharey=True)
# ax11.plot([1,2],[1,2])
#
# # plt.tight_layout()

fig=plt.figure()
x=[1,2,3,4,5,6,7]
y=[1,3,4,2,5,8,6]

# 这里是相对于figure左边10%，底部10%做起点，宽度占figure大小的80%，高度占figure大小的80%
left,bottom,width,height=0.1,0.1,0.8,0.8
ax1=fig.add_axes([left,bottom,width,height])

ax1.plot(x,y,'r')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('title')


left,bottom,width,height=0.2,0.6,0.25,0.25
ax1=fig.add_axes([left,bottom,width,height])

ax1.plot(x,y,'b')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('title inside 1')

# 此法和第二步没区别
plt.axes([0.6,0.2,0.25,0.25])
# 这里先将y逆序，
plt.plot(y[::-1],x,'g')
plt.xlabel('x')
plt.ylabel('y')
plt.title('title inside 2')

plt.show()


```

![1678957310271](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1678957310271.png)

##### 15.次坐标轴

```python
import matplotlib.pyplot as plt
import numpy as np

x=np.arange(0,10,0.1)

y1=0.05*x**2
y2=-1*y1

fig,ax1=plt.subplots()

# 翻转y轴，x轴共用
ax2=ax1.twinx()
ax1.plot(x,y1,'g-')
ax2.plot(x,y2,'b--')

ax1.set_xlabel('X data')
ax1.set_ylabel('Y1',color='g')
ax2.set_ylabel('Y2',color='g')

plt.show()
```

![1678962872705](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1678962872705.png)

##### 16.Animation动画

```python
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

fig,ax=plt.subplots()


x=np.arange(0,2*np.pi,0.01)

line,=ax.plot(x,np.sin(x))

def animate(i):
    line.set_ydata(np.sin(x+i/10))
    return line,

def init():
    line.set_ydata(np.sin(x))
    return line,

ani=animation.FuncAnimation(fig=fig,func=animate,frames=100,init_func=init,interval=20,blit=False)
plt.show()
```

![1678963799885](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1678963799885.png)



##### 17.总结

先用一个空list来装载需要绘制的数据，然后再将需要绘制的数据绘图出来即可