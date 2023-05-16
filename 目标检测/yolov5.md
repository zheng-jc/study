#### yolov5 5.0

首先在github上下载yolov5的5.0版本

YOLOv5官方项目地址：https://github.com/ultralytics/yolov5

点击target找到5.0版本直接下载code的zip包就可以了

![1683363402161](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683363402161.png)



下载好后解压，用pycharm打开一个新的项目

![1683363484280](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683363484280.png)

安装好所需呀的包，运行detect.py文件就可以了



#### 报错

运行出现错误

![1683363573649](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683363573649.png)

直接在github上下载好它对应的模型就可以了

先找到5.0版本，在这个目录下点击找到对应版本就好了

![1683363702936](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683363702936.png)

![1683363810602](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683363810602.png)

复制对应模型的下载链接，使用迅雷下载，最后将下载好的模型复制到yolov5项目的根目录下就行

运行还是报错

![1683364239546](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683364239546.png)

报SPPF错误的是因为自动下载下来的权重文件是最新的，需要自己去下载对应tag的权重



出现错误的原因：由于yolov5目前最新版本为v6.1，但我跑的是5.0版本，则运行detect.py时自动从github上下载的训练好的模型为最新版本v6.1。从而导致运行环境和模型版本不一致，从而报错。

一、AttributeError: Can‘t get attribute ‘SPPF‘ on ＜module ‘models.common‘ from ‘H:\\yolov5-5.0\\models\\

二、yolov5 ERROR: AttributeError: ‘Upsample‘ object has no attribute ‘recompute_scale_factor‘

三、yolov5中The size of tensor a (80) must match the size of tensor b (56) at non-singleton dimension 3

四、UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  C:\cb\pytorch_1000000000000\work\aten\src\ATen\native\TensorShape.cpp:2228.)  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]





一、AttributeError: Can‘t get attribute ‘SPPF‘ on ＜module ‘models.common‘ from ‘H:\\yolov5-5.0\\models\\



解决方案：在新版本的models/common.py里面去找到这个SPPF的类,把它拷过来到你的models/common.py里面,这样你的代码就也有这个类了,还要引入一个warnings包就行了！还要引入一个warnings包就行了！

有的同学找不到SPPF这个类，那我现在直接粘贴在这里，你们只需要复制到你们的common.py里面即可，记得把import warnings放在上面去：

``` python
import warnings

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))
```

![1683367829117](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683367829117.png)



二、yolov5 ERROR: AttributeError: ‘Upsample‘ object has no attribute ‘recompute_scale_factor‘

解决方案:

找到\torch\nn\modules\upsampling.py下的文件

import torch.nn.modules.upsampling，然后摁住ctrl+鼠标左键就会跳转到该文件下，或者摁提示报错的地方也可以跳转到该文件下

之后将如下图所示154行注释掉就行了

![1683367234205](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683367234205.png)

![img](http://zhangshiyu.com/zb_users/upload/2022/12/20221224123302167185638223786.png)



三、yolov5中The size of tensor a (80) must match the size of tensor b (56) at non-singleton dimension 3



下载：
https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt

替换默认下载的yolov5s.pt，因为默认下载的是V6.1的，这里发现之前自己手动下载对应的模型居然不行，点击上面那个链接下载又可以

替换后，在运行 detect.py就OK了

![1683367380491](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683367380491.png)

四、UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  C:\cb\pytorch_1000000000000\work\aten\src\ATen\native\TensorShape.cpp:2228.)  return _VF.meshgrid(tensors, \**kwargs)  # type: ignore[attr-defined]

 找到functional.py的第568行，

将 return _VF.meshgrid(tensors, **kwargs）改为 return _VF.meshgrid(tensors, **kwargs,indexing='ij')

![1683367520209](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683367520209.png)

![1683367582231](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683367582231.png)



#### 完美运行

![1683367939115](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683367939115.png)



对应运行结果存放的照片，点击就可以查看结果了

![1683367998991](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683367998991.png)

![1683368069522](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683368069522.png)



#### 参数查看及修改

![1683368158037](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683368158037.png)





--source

默认将data/image下的所有图片进行检测，并把结果保存起来

这里也可以换成某个具体图片的路径

或者换成视频

手机下载ip摄像头，路径改为http://admin:密码@手机上打开之后显示的ip地址，就可以实时检测，不为rtsp格式也可以



-- image-size

对应网络匹配的一个图像大小输入

![1683369401155](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683369401155.png)



--conf-thres

置信度，默认置信度大于0.25的时候才会认为它是一个物体（类别），或者说才会被识别



--iou-thres

iou是交并比

通俗来说iou的大小阈值用来确定这多个框是不是检测的是同一个物体，如果是，通过非最大值抑制ＮＭＳ框处最大概率的那个框

设置成0，那检测出来的框不会有交集





NMS步骤：

第一步：对 BBox 按置信度排序，选取置信度最高的 BBox（所以一开始置信度最高的 BBox 一定会被留下来）；

第二步：对剩下的 BBox 和已经选取的 BBox 计算 IOU，淘汰（抑制） IOU 大于设定阈值的 BBox（在图例中这些淘汰的 BBox 的置信度被设定为0）。

第三步：重复上述两个步骤，直到所有的 BBox 都被处理完，这时候每一轮选取的 BBox 就是最后结果。


![1683438871511](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683438871511.png)

在上面这个例子中，NMS只运行了两轮就选取出最终结果：第一轮选择了红色BBox，淘汰了粉色BBox；第二轮选择了黄色BBox，淘汰了紫色 BBox和青色BBox。注意到这里设定的IoU阈值是0.5，假设将阈值提高为0.7，结果又是如何？

![1683438936755](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683438936755.png)

可以看到，NMS 用了更多轮次来确定最终结果，并且最终结果保留了更多的 BBox，但结果并不是我们想要的。因此，在使用 NMS 时，IoU 阈值的确定是比较重要的，但一开始我们可以选定 default 值（论文使用的值）进行尝试。







--device

这个参数意思就是指定GPU数量，如果不指定的话，他会自动检测，默认是空，也就是cpu





action='store_true'

你指定了这个参数后，他就会有一个相应的动作

![1683370133201](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683370133201.png)



![1683370107826](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683370107826.png)

--view-img

指定这个参数后，程序在运行的时候会显示图片，然后再关掉

比如运行视频文件的时候，就可以实时看到画面检测的的效果了



--save-txt

保存结果为txt



--nosave

即不保存检测的图片结果，开启这个参数就是不保存预测的结果，但是还会生成exp文件夹，只不过是一个空的exp



--classes

只保留某一个类别的检测结果

比如，--classes 0，只保留0类的检测结果，也就是只保留预测结果是人的检测框



--augment

是否使用数据增强，开启后，预测置信度会提高



--project

检测结果保存的位置，默认是保存在runs/detect，也可以进行修改

--name

指定保存图片的文件夹的名字，默认的exp



--exist-ok

如果开启的话，那么每次运行就不会将结果保存在新文件夹里了，而是保存name指定的文件夹中，即在exp里面，再次运行会覆盖原来的结果



这些参数会在opt这个对象里，也可以在这一行的代码里打断点，看里面的默认值的情况





#### 调用电脑摄像头

将--source里的默认参数改成'0'，后还是报错

![1683437152976](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683437152976.png)



![1683437184214](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683437184214.png)



![1683436435149](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683436435149.png)



我们需要在datasets.py文件中找到下面这行代码：

![1683436646218](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683436646218.png)



改成

![1683436916005](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683436916005.png)





第二处

![1683436992929](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683436992929.png)





![1683437077584](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683437077584.png)



现在问题是，能运行并且能调用到摄像头，摄像头的灯是亮的，但是没有图像显示

![1683437618996](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683437618996.png)



就是在utils/general.py 文件中有一个is_docker()函数
返回修改为注释的内容，如下(这是修改之前的)，修改之后返回路径是.dockerenv

改之前

![1683437974353](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683437974353.png)

改之后

![1683438053877](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683438053877.png)



再次运行detect.py

完美显示了

![1683438186575](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683438186575.png)



#### 利用云服务器训练自己的yolov5

利用google的colab

首先将项目压缩，然后将压缩文件上传到colab上，然后使用命令行对压缩包进行解压

![1683513249618](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683513249618.png)

![1683513418345](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683513418345.png)

```python
!unzip /content/yolov5-5.0 -d /content/yolov5
```



发现多了一个没用的文件夹，手动删除不了

![1683513660459](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683513660459.png)

使用命令行删除

```python
！rm -rf /content/yolov5/__MACOSX
```



然后cd到yolov5-5.0这个文件夹下面

```python
%cd /content/yolov5/yolov5-5.0
```



安装程序运行需要的一些包

```python
!pip install -r requirement.txt
```

![1683514073273](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683514073273.png)



也可以使用tensorboard，要先加载一下插件

```python
%load_ext tensorboard
```

先开启tensorboard

```python
%tensorboard --logdir=runs/train
```

![1683514586658](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683514586658.png)

运行train.py文件，并启动矩阵推理

```python
!python train.py --rect
```

![1683514704993](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683514704993.png)

如果要使用完整的coco数据集进行训练

```python
!python train.py --rect --data=data/coco.yaml
```

![1683517033947](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683517033947.png)

测试的话可以使用训练的那个best.pt，复制它的路径，到detect.py那个权重weights默认的那个模型





#### 制作和训练自己的数据集

##### 标注

yolov5那里也有[教程](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#13-prepare-dataset-for-yolov5)



![1683633817429](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683633817429.png)

先创建一个yaml文件

![1683634228670](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683634228670.png)

找好图片，在[make sense](https://www.makesense.ai)在线网站上打标签

先将找好的图片添加进去，然后会询问是目标检测还是图像识别任务

![1683634435470](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683634435470.png)

开始之前，会询问你是否从一个txt文件中添加标签



![1683634588926](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683634588926.png)

每个类别写在一行里面，这样你上传文件的时候，就会自动识别你这个软件

![1683634685672](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683634685672.png)

上传完之后，就可以开始项目了

![1683634840973](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683634840973.png)

为了加快打标签的速度，可以使用现有的模型

![1683634928470](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683634928470.png)

![1683634948901](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683634948901.png)



模型它说发现了一些类别，询问你是否添加到label里

![1683635031453](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683635031453.png)



把光标移动到左上角，添加标签就可以了

![1683635098190](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683635098190.png)





#### yolov3

将yolov3部署到自己的pycharm后，运行detect.py文件后，会报错说要下载Arial.ttf到你电脑对应的文件夹里，这里可以手动将该文件下载到指定的目录里就可以了



好像还有一个错误，修改这里就行了

![1683700231558](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683700231558.png)

还有一个问题是，同样将检测源改为0时，可以运行成功，但是不出现图像，解决方法，同上

![1683700163087](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683700163087.png)

#### yolov5 vs yolov3

同样是打开摄像头，yolov5 5.0比yolov3 9.6快了差不多四倍

![1683699307014](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683699307014.png)



![1683699351216](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1683699351216.png)