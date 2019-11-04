


# 模型设计技术概述
## 深度学习兴起前的网络
- neocognitron: 真正意义上的第一个卷积网络
- TDNN （Time delay neural network）：第一个用于语音的卷积网络，输入声音频谱图
- Cresceprton： cresco(grow)and perceptio(percerption)，即分层机制
- Boltzmann machine：包含2个网络层，一个可见层，一个隐藏层。本质上属于随机神经网络和递归神经网络。增加隐藏层数量，得到深度玻尔兹曼机(DBM)，是无向的。如果出了顶层的隐藏层连接，其他层连接都是有向的，就得到深度信念网络（DBN)。DBN与DBM不同，也与当前的DNN不同，它的最顶层包含了一个无向层，却限制了其他底层节点的方向，本质上就是用于将数据层（可视层）进行降维。
## 2D/3D经典网络
**2D**：

- LeNets ：CNN的“Hello World”，反向传播理论的应用
- AlexNet： 比LeNets更深更宽，引入了**ReLU激活函数**、**LRN归一化**、**Dropout**等训练方法
- ZfNet：验证了网络分层机制（抽象层级逐渐提高），大卷积和步长会带来负面影响
- VGG：增加了AlexNet的深度，性能变好。但是网络深度增加使优化困难，VGG19训练效果不如VGG16
- 1*1卷积：它的大小是`1*1`，没有考虑在前一层局部信息之间的关系。最早出现在 Network In Network的论文中。使用`1*1`卷积可以加深网络，提高网络表征能力。在Inception网络中用来降维。

**3D**：