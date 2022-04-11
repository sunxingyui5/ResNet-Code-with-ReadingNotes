## Deep residual learning for image recognition  
注：配合阅读[Identity mappings in deep residual networks](https://readpaper.com/paper/2949427019)  
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
·假设H(x)为最终要学习的映射，x是输入，让网络拟合F(x)=H(x)-x  
·如果卷积层后加Batch Normalization层，则不需要偏置项
·残差F(x)与自身输入x维度必须一致才能实现逐元素相加  
·残差可以表现为多层的CNN，逐元素加法可以表现为两个feature maps逐通道相加  
### 本质  
传统多层网络难以准确拟合，加了恒等映射后，深的网络不会比浅层网络的效果更差，如果恒等映射足够好，可以把所有的权重都学成0  
### 对比实验  
普通无残差： 类似VGG，每个block内filter数不变，feature map大小减半时filter个数x2，用步长为2的卷积执行下采样，Global Average Pooling取代全连接层，更少的参数和计算量防止过拟合  
残差网络：实线代表维度相同的直接相加，代表出现了下采样，即步长为2的卷积  
**残差分支出现下采样时：**
>对于<font color="red">shortcut connection：</font>  
**A方案：** 多出来的通道<font color="red">padding</font>补0填充  
**B方案：** 用<font color="red">$1\times 1卷积$</font>升维  

不管采取那种匹配维度方案，<font color="red">shortcut</font>分支第一个卷积层步长都为<font color="red">2</font>  
![Controlexperiment](https://github.com/sunxingyui5/ResNet-Code-with-ReadingNotes/blob/main/img/ControlExperiment.png)  
### 训练  
·和<font color="red">Alex Net</font>，<font color="red">VGG</font>遵循一样的范式  
<font color="red">图像增强：</font>随机被压缩到<font color="red">[256,480]</font>之间，做尺度增强，用<font color="red">$224\times 224$</font>随机截取小图，再做<font color="red">水平镜像</font>作为图像的增强  
·在每一个<font color="red">卷积层</font>后面和<font color="red">激活</font>之前，都使用一个<font color="red">Batch Normalization</font>（<font color="red">BN-Inception</font>提出）
>遵循<font color="red">PReLU</font>的权重初始化方法
>
><font color="red">SGD</font>的<font color="red">batch</font>是<font color="red">256</font>
>
><font color="red">lr</font>开始是<font color="red">0.1</font>，遇到瓶颈<font color="red">${\div}10$</font>
>
><font color="red">$L_2$正则化</font>为<font color="red">0.0001</font>
>
><font color="red">动量</font>是<font color="red">0.9</font>  
>
>没有使用<font color="red">dropout</font>（BN和dropout不共存）  
### 测试  
遵循<font color="red">Alex Net</font>的<font color="red">10-crop testing</font>（一张图片裁成10个小图分别喂入网络，再汇总）  
<font color="red">fully convolution form</font>把图片缩放到不同的尺寸，再对不同尺度结果融合  
**注：**  
$$ \left\{
\begin{matrix}
 数据本身决定了该类问题的上限，而模型（算法）只是逼近这个上限 \\
    \\
 模型结构本身决定了模型的上限，而训练调参只是在逼近这个上限 
\end{matrix}
\right.
$$   
·梯度可以通过<font color="red">shortcut connection</font>传回到底层，所以不会出现层层盘剥  
### 三个残差网络方案  
**A：所有<font color="red">shortcut</font>无额外参数，升维时用<font color="red">padding</font>补0  
B：平常的<font color="red">shortcut</font>用<font color="red">identity mapping</font>，升维时用<font color="red">$1 \times 1$卷积</font>升维  
C：所有<font color="red">shortcut</font>都使用<font color="red">$1 \times 1$卷积
   </font>**  
**结果：**  
$$ \left\{
\begin{matrix}
 B比A好：A在升维时用padding补0相当于丢失了shortcut分支的信息，没有进行残差学习 \\
    \\
 C比B好：C的13个下采样残差模块的shortcut都有参数，模型表示能力强  \\
    \\
 ABC差不多：说明identity mapping的shortcut足已解决退化问题
\end{matrix}
\right.
$$   
### Bottleneck（瓶颈）结构  
·无参数<font color="red">identity shortcuts</font>（联接两个高维端）对<font color="red">bottleneck</font>结构而言是十分重要的（如果换成其它映射，那么时间复杂度、计算量、模型尺寸、参数量都会翻倍）  
·模型层数为“<font color="red">带权重的层数</font>”（如<font color="red">pooling</font>和<font color="red">softmasx</font>不算）  
![bottleneckblock](https://github.com/sunxingyui5/ResNet-Code-with-ReadingNotes/blob/main/img/bottleneckblock.jpg)  
$$ basic bolck\left\{
\begin{matrix}
 ResNet-18 \\
    \\
 ResNet-34 
\end{matrix}
\right.
$$
    $$ bottleneck block\left\{
\begin{matrix}
 ResNet-50 \\
 ResNet-101   \\
 ResNet-152 
\end{matrix}
\right.
$$  
### ResNet-50  
将<font color="red">ResNet-34</font>中只有两层的残差模块换成了3层的<font color="red">bottleneck</font>残差模块，变成了<font color="red">50层</font>，下采样中用<font color="red">B方案</font>（平常的<font color="red">shortcut</font>用<font color="red">identity mapping</font>下采样时用<font color="red">$1 \times 1$卷积</font>）  
### ResNet-101&ResNet-152  
用<font color="red">3层bottleneck block</font>构建，计算量低与<font color="red">VGG-16/19</font>  
更深的层，更加准确
