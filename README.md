caffe一键式训练评估集成开发环境
====================================

#### 深度学习入门进阶指南

#### Last Update 2017.09.19

<p align="left">
    <img src="http://www.tendyron.com/images/logo.gif">
</p>

## 概述

深度学习大火以来，各类教程充斥网络、博客等，甚至各类培训也屡见不鲜，但无不例外均存在一个问题就是各个环节缺乏联系，没有一个统一的开发环境去完成整个流程。这个项目的目的就是提供一个集成式开发环境，大大提升开发部署的效率。

使用深度学习完成一个特定的任务比如说字符识别、人脸识别等大致可以分为数据准备、定义模型、训练模型、评估模型和部署模型等几个步骤。

### 1.数据准备

首先收集要任务相关的数据，这里准备了一个车牌字符数据（仅包含0-9共10个数字），直接解压[data.rar](data.rar)到当前文件夹即可，格式如下图所示，每类图片对应一个文件夹，放到一个data文件夹下，注意格式一致型（都为.jpg或.png文件），仔细筛查，不要含有其他的非图片文件在里面，你也可以用自己的数据替换这些车牌字符数据。

![图片整理方式](doc/data.png)

caffe使用了lmdb内存数据库等来加快训练时读取数据的速度，为此，caffe自带的tools里提供了一个工具（可由convert_imageset.cpp编译生成），它的输入是图片路径和标签对组成的文件，每次都手动生成这个文件不胜其烦。

我们希望是自动化的从文件夹读取的功能，因此，本项目通过preprocess/preprocess.py来获取如下图所示的文件夹下所有的文件路径以及对应的文件标签的功能，它输出训练和验证集preprocess/train.txt和preprocess/val.txt以及标签映射文件modef/labels.txt

python util/preprocess.py
然后用以下代码转换成lmdb数据库

```
del "lmdb/train_lmdb\*.*" /f /s /Y
del "lmdb/val_lmdb\*.*" /f /s /Y
rd /s /q "lmdb/train_lmdb"
rd /s /q "lmdb/val_lmdb"

"../build/tools/convert_imageset" --resize_height=20 --resize_width=20 --shuffle "" "preprocess/train.txt" "lmdb/train_lmdb"

echo "Creating val lmdb..."
"../build/tools/convert_imageset" --resize_height=20 --resize_width=20 --shuffle "" "preprocess/val.txt" "lmdb/val_lmdb"
```

### 2.定义模型

训练定义文件位于modeldef下的plate_train_test.prototxt，部署文件在deploy.prototxt，你可以通过[网络结构可视化](http://ethereon.github.io/netscope/#/editor)对这些网络进行可视化，以便更清晰的理解他们的含义。

### 3.训练模型

Train.bat训练模型使用的是如下命令：

```
"../build/tools/caffe" train --solver=modeldef/solver.prototxt
```

### 4.评估模型

evaluation.bat用来对data文件下下的数据进行评估，它会得出迭代次数为10000时模型的错误率，并且打印出误识别图片对应的真值和预测值，并把相应数据保存在error文件夹下，命名格式为字符文件夹/图片在文件夹内的序号_真值类别_预测类别(以0/190_0_4.jpg为例，代表0/190.jpg被误识为4)，这些错误识别的样本需要仔细分析，不断调试参数，以获得期望的结果。

<p align="center">
    <img src="figures/error.png", width="600">
</p>

不难看出，本项目提供的训练好的[模型文件](trainedmodels/platerecognition_iter_10000.caffemodel)错误率低于0.6%,这就意味着其达到了99.4%以上的准确率。

```
以上4个步骤可以通过一个脚本一键式完成
本项目提供了oneclick.bat来完成一键式生成数据、训练和评估模型，大大提升了开发效率。
```

### 5.部署模型

由于速度原因，实际中多使用C++而不是python进行部署，因此本项目在cpp文件夹下提供了evaluationcpp工程，它使用单例模式来防止每次预测都加载模型，只需使用如下代码即可在你的项目中一行代码使用CNN，此外，该项目也提供了对模型进行评估的功能。

```
 cv::Mat img=cv::imread("imagepath.jpg");
 string result=CnnPredictor::getInstance()->predict(img);
```

当然，你也可以运行calssification.bat来调用caffe自身进行分类识别

```
"../build/examples/cpp_classification/classification" "modeldef/deploy.prototxt" "trainedmodels/platerecognition_iter_1000.caffemodel" "modeldef/mean.binaryproto" "modeldef/labels.txt" "data/0/4-3.jpg"
```
<p align="center">
    <img src="doc/classification.png", width="600">
</p>

其返回了最高的5个类别的相似度，不难看出训练的网络对于data/0/0.jpg有高达93%的概率认为其属于0这个字符，结果还是非常理想的

## 环境搭建

* [官方Caffe-windows 配置与示例运行](http://blog.csdn.net/guoyk1990/article/details/52909864)

* [Ubuntu16.04+cuda8.0+caffe安装教程](http://blog.csdn.net/autocyz/article/details/52299889)

* [如何快糙好猛地在Windows下编译CAFFE并使用其matlab和python接口](http://blog.csdn.net/happynear/article/details/45372231)

## 实现解析

* [网络结构可视化](http://ethereon.github.io/netscope/#/editor)

* [图文并解caffe源码](http://blog.csdn.net/mounty_fsc/article/category/6136645)

* [caffe源码解析](http://blog.csdn.net/qq_16055159)

* [caffe代码阅读](http://blog.csdn.net/xizero00/article/category/5619855/)

* [从零开始山寨Caffe caffe为什么这么设计？](http://www.cnblogs.com/neopenx/)

* [Caffe代码导读 21天实战caffe作者博客](http://blog.csdn.net/kkk584520/article/category/2620891/2)

* [CNN卷积神经网络推导和实现](http://blog.csdn.net/zouxy09/article/details/9993371)

* [（Caffe）卷积的实现](http://blog.csdn.net/mounty_fsc/article/details/51290446)

* [（Caffe，LeNet）反向传播（六）](http://blog.csdn.net/mounty_fsc/article/details/51379395)

* [caffe卷积层代码阅读笔记](http://blog.csdn.net/tangwei2014/article/details/47730797)

* [caffe的python接口学习](http://www.cnblogs.com/denny402/tag/caffe/default.html?page=2)

## 添加新层及样例解析

* [Caffe 增加自定义 Layer 及其 ProtoBuffer 参数](http://blog.csdn.net/kkk584520/article/details/52721838)

* [caffe添加新层教程](http://blog.csdn.net/shuzfan/article/details/51322976)

* [caffe自带样例解析](http://blog.csdn.net/whiteinblue)

* [深度卷积网络CNN与图像语义分割](http://blog.csdn.net/xiahouzuoxin/article/details/47789361)

* [caffe特征可视化的代码样例](http://blog.csdn.net/lingerlanlan/article/details/37593837)

## 技术交流QQ群

* 本项目专用群:238787044

<p align="left">
<img src="http://i.imgur.com/7cjLpED.png", width="200">
</p>

* Caffe 深度学习交流群：534492004

* MXNet 深度学习交流群：489170253

## 参考：

* [mxnet 训练自己的数据](https://github.com/imistyrain/mxnet-mr)

* [MatconvNet 训练自己的数据](https://github.com/imistyrain/MatConvNet-mr)

如果此项目对您有用，请在能力所及范围支持开源事业的发展，扫码微信打赏
<center  class="half">
<img src="http://i.imgur.com/fkA4iPX.jpg", width="200"><img src="http://i.imgur.com/tYLAMg3.jpg", width="200">
</center>