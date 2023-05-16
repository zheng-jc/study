**在工作中，大部分都是使用VScode编写代码，并通过SSH远程连接服务器，时刻将代码和数据放在服务器上。**

在进行图像处理和跑深度学习模型过程中，想在本地查看服务器上的图片处理效果（im.show()）或模型指标变化结果图片（plt.scatter()等），是没办法使用这些指令（im.show()、plt.scatter()），在本地桌面看图片的。

<font color='red'>不过有个方法，可以把图片保存下来，再使用VScode打开查看，下面记录一下VSCode+SSH环境下显示matplotlib绘制的figure</font>



#### 1.代码

plt.switch_backend('agg') 必须加上这行代码，不然不能将图片保存到centos

```python
import numpy as np
import matplotlib.pyplot as plt
 
plt.switch_backend('agg')
from sklearn.datasets.samples_generator import make_blobs
 
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.2, 0.2, 0.2], 
                  random_state =9)
plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.savefig("./1.jpg")
```



#### 2.图片显示

![1682583570359](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1682583570359.png)