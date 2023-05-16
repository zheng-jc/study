首先你的有梯子，还得有谷歌账号，然后在google浏览器中输入https://colab.research.google.com/

看到如下界面，新建笔记本即可，(我是因为有了谷歌账号，右上角显示账号已登录)



![1679732416323](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1679732416323.png)



![1679732500601](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1679732500601.png)

这里的笔记本的使用方法和jupyter是一样的

如果要使用gpu训练，点击修改里的笔记本设置，将None改成GPU即可，就可以使用gpu训练啦

![1679732645928](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1679732645928.png)

![1679732704271](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1679732704271.png)



这里可以查看一下google提供的gpu的配置，可以看到是一个特斯拉T4的差不多16G显存的显卡

> !nvidia-smi

![1679733061392](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1679733061392.png)



然后将要运行的代码复制到这个新的代码框中

![1679732072109](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1679732072109.png)



点击左上角的运行按钮，等待运行结果即可

![1679732168467](C:\Users\郑金财\AppData\Roaming\Typora\typora-user-images\1679732168467.png)

