## Deep residual learning for image recognition  
**阅读地址：** [Deep residual learning for image recognition](https://readpaper.com/paper/2949650786)

**知乎精讲：** [ResNet 精读笔记](https://zhuanlan.zhihu.com/p/496445232)

**推荐学习视频：** [ResNet论文逐段精读【论文精读】](https://www.bilibili.com/video/BV1P3411y7nn/?spm_id_from=333.788)&[【精读AI论文】ResNet深度残差网络](https://www.bilibili.com/video/BV1vb4y1k7BV?p=4)

**被引用次数：** 112838（截至2022.04.11）

**官方开源：** [pytorch_ResNet](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)

**注：** 建议配合食用[Identity Mappings in Deep Residual Networks](https://readpaper.com/paper/2949427019)

### 提供的思路  
神经网络不需要去拟合复杂的底层映射了，只需要拟合在原来输入的基础上要进行哪些偏移，哪些修改，最总只要拟合残差就好了  
这样使深的网络不会比浅层网络效果更差，最多只会让后续网络变为恒等映射  
![CNNs](https://github.com/sunxingyui5/ResNet-Code-with-ReadingNotes/blob/main/img/CNNs.png)  
### 提出残差学习结构解决非常深网络的退化问题和训练问题  
·每层都学习相对于本层输入的残差，然后与本层输入加法求和，残差学习可以加快优化网络，加深层数，提高准确度  
·直接将网络堆深  
>①梯度消失/梯度爆炸：阻碍收敛（现可以通过初始化权重解决）
>
>②网络退化：不是任何网络都能被相同的优化

·不拟合底层，拟合残差（如果恒等映射足够好，可以把所有权重都学成0）  
·本文中shortcut connection只用来进行恒等映射，不引入额外的参数量和计算量（加法计算几乎可以忽略）  
·门控函数“highway networks”扮演残差角色，但深层网络性能提升不明显  
    
### 如何防止梯度消失？  
初始化和Batch Normalization，通过SGD和反向传播就开始收敛了
浅模型输入 = 浅模型 + 输入不变 = 汇总输出 （递归结构难以被优化）)

### 残差学习  
·假设![](http://latex.codecogs.com/svg.latex?H\\(x\\) )为最终要学习的映射，![](http://latex.codecogs.com/svg.latex?x)是输入，让网络拟合![](http://latex.codecogs.com/svg.latex?F\\(x\\)=H\\(x\\)-x ) 
·如果卷积层后加Batch Normalization层，则不需要偏置项
·残差![](http://latex.codecogs.com/svg.latex?F\\(x\\))与自身输入![](http://latex.codecogs.com/svg.latex?x)维度必须一致才能实现逐元素相加  
·残差可以表现为多层的CNN，逐元素加法可以表现为两个feature maps逐通道相加  
### 本质  
传统多层网络难以准确拟合，加了恒等映射后，深的网络不会比浅层网络的效果更差，如果恒等映射足够好，可以把所有的权重都学成0  
### 对比实验  
普通无残差： 类似VGG，每个block内filter数不变，feature map大小减半时filter个数x![](http://latex.codecogs.com/svg.latex?2)，用步长为![](http://latex.codecogs.com/svg.latex?2)的卷积执行下采样，Global Average Pooling取代全连接层，更少的参数和计算量防止过拟合  
残差网络：实线代表维度相同的直接相加，代表出现了下采样，即步长为![](http://latex.codecogs.com/svg.latex?2)的卷积  
**残差分支出现下采样时：**
>对于shortcut connection：  
**A方案：** 多出来的通道padding补![](http://latex.codecogs.com/svg.latex?0)填充  
**B方案：** 用1x1卷积升维  

不管采取那种匹配维度方案，shortcut分支第一个卷积层步长都为![](http://latex.codecogs.com/svg.latex?2)  
![Controlexperiment](https://github.com/sunxingyui5/ResNet-Code-with-ReadingNotes/blob/main/img/ControlExperiment.png)  
### 训练  
·和Alex Net，VGG遵循一样的范式  
**图像增强：** 随机被压缩到![](http://latex.codecogs.com/svg.latex?[256,480])之间，做尺度增强，用![](http://latex.codecogs.com/svg.latex?224\times 224)随机截取小图，再做水平镜像作为图像的增强  
·在每一个卷积层后面和激活之前，都使用一个Batch Normalization（BN-Inception提出）
>遵循PReLU的权重初始化方法
>
>SGD的batch是![](http://latex.codecogs.com/svg.latex?256)
>
>lr开始是![](http://latex.codecogs.com/svg.latex?0.1)，遇到瓶颈![](http://latex.codecogs.com/svg.latex?{\div}10)
>
>![](http://latex.codecogs.com/svg.latex?L_2)正则化为![](http://latex.codecogs.com/svg.latex?0.0001)
>
>动量是![](http://latex.codecogs.com/svg.latex?0.9)
>
>没有使用dropout（BN和dropout不共存）  

### 测试  
遵循Alex Net的10-crop testing（一张图片裁成10个小图分别喂入网络，再汇总）  
fully convolution form把图片缩放到不同的尺寸，再对不同尺度结果融合  
**注：**  
>数据本身决定了该类问题的上限，而模型（算法）只是逼近这个上限 
>    
>模型结构本身决定了模型的上限，而训练调参只是在逼近这个上限 

·梯度可以通过shortcut connection传回到底层，所以不会出现层层盘剥  

### 三个残差网络方案  
>A：所有shortcut无额外参数，升维时用padding补0   
>B：平常的shortcut用identity mapping，升维时用1x1卷积升维   
>C：所有shortcut都使用1x1卷积 

**结果：**  
>B比A好：A在升维时用padding补0相当于丢失了shortcut分支的信息，没有进行残差学习 
>   
> C比B好：C的13个下采样残差模块的shortcut都有参数，模型表示能力强  
>    
> ABC差不多：说明identity mapping的shortcut足已解决退化问题
   
### Bottleneck（瓶颈）结构  
·无参数identity shortcuts（联接两个高维端）对bottleneck结构而言是十分重要的（如果换成其它映射，那么时间复杂度、计算量、模型尺寸、参数量都会翻倍）  
·模型层数为“带权重的层数”（如pooling和softmasx不算）  
![bottleneckblock](https://github.com/sunxingyui5/ResNet-Code-with-ReadingNotes/blob/main/img/bottleneckblock.jpg)  
#### basic bolck
> ResNet-18 
>   
> ResNet-34 

#### bottleneck block
> ResNet-50 
> ResNet-101   
> ResNet-152 
 
### ResNet-50  
将ResNet-34中只有两层的残差模块换成了3层的bottleneck残差模块，变成了50层，下采样中用B方案（平常的shortcut用identity mapping下采样时用1x1卷积）  
### ResNet-101&ResNet-152  
用3层bottleneck block构建，计算量低与VGG-16/19 
更深的层，更加准确
