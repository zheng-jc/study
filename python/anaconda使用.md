**注：**在[Anaconda安装](https://so.csdn.net/so/search?q=Anaconda安装&spm=1001.2101.3001.7020)的过程中，**比较容易出错的环节是环境变量的配置，**所以大家在配置环境变量的时候，要细心一些。

**步骤一：**输入链接“https://www.anaconda.com/”登录Anaconda官网。下载对应版本

![1676951919052](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1676951919052.png)

![1676952040895](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1676952040895.png)

![1676952067062](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1676952067062.png)

![1676952133256](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1676952133256.png)

注意这里的安装目录不要带有中文，建议安装在d盘

![1676958015581](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1676958015581.png)

![1676958462303](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1676958462303.png)

![1676958442184](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1676958442184.png)

更换一个无空格的安装路径即可

![1676958519664](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1676958519664.png)

点击next，默认即可





查看显卡，主页搜索设备管理器，查看显示适配器

![1676959337417](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1676959337417.png)

或者电脑win+R键打开命令行窗口，然后输入nvidia-smi，检查CUDA Version这个版本号

```bash
nvidia-smi
```

会看到如下页面显示

![1667032621734](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1667032621734.png)



然后进入pytorch官网，选择对应版本的安装方式即可

![1676959702949](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1676959702949.png)



#### 3.有序的管理环境

不同的项目，需要不同版本的环境，比如这个项目要用到pytorch0.4，另一个项目要用到pytorch1.0，你可以创造两个屋子，一个屋子放0.4版本，另一个屋子放1.0版本，你需要哪个版本，就进哪个屋子工作，我们首先使用conda指令创造一个屋子，叫做pytorch

指令如下

> conda create -n pytorch python=3.6

conda是指调用conda包，create是创建的意思，-n是指后面的名字是屋子的名字，pytorch是屋子的名字（这个可以任意），python=3.6是指创建的屋子是python3.6版本

![1676961560767](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1676961560767.png)

安装成功，如果要激活这个环境的话，使用下面这条指令conda  activate pytorch

![1676961664561](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1676961664561.png)

进入这个屋子，查看里面有什么包

![1676961834322](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1676961834322.png)

安装pytorch

> ```
> conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
> ```

![1676962163104](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1676962163104.png)

这个方法安装巨慢

使用国内镜像下载

https://mirrors.bfsu.edu.cn/anaconda/cloud/pytorch/linux-64/

在该房子里添加如下代码

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
```

然后去pytorch官网里复制那行代码，只复制到版本号那里

![1676966662774](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1676966662774.png)



![1676966776911](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1676966776911.png)

等待安装完成即可





移除当前房子conda remove -n pytorch --all

查看所有python环境conda info --env

