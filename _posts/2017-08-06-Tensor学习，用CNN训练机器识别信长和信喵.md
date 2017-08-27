---
title: TensorFlow学习——用CNN训练机器识别信长和信喵
tag: tech
---


最近想搞个图片分类器，实现自己加载本地的图片进行训练，然后保存模型，另起一个程序加载模型，然后读入几张图片进行预测。回头盘点了下几个熟悉开源的DL工具，觉得做图片分类还是tensorflow比较方便，于是就找了点图片完成了这个模型，这里记录一下。

## 一.数据准备
用什么数据来构造分类器呢？记得前两年很喜欢玩《信长之野望14》，里面不是有很多人物头像吗？而且还分``信长之野望原版``和``喵之信长``版，索性做个分类器识别是人头还是猫头好了。      

数据下载地址（百度云）：
> 链接: https://pan.baidu.com/s/1slUDK0t 密码: vjv8

这里面将头像数据分别放在了两个目录下，
```
nobunaga ：原版头像
nobunyaga ：喵版头像
```
我从他们里面分别把信长的头像取了出去不做训练，看看最后的模型是否能够认识他。

## 二.载入数据
首先，用什么方法读取图片呢？百度一下之后觉得PIL是比较方便的，于是用pip装了下PIL就可以使用了。在pycharm中新建一个工程，创建python脚本CnnClassify，第一部分代码编写如下：
```
# -*- coding: utf-8 -*-

from PIL import Image
import glob
import os
import tensorflow as tf
import numpy as np
import time

img_path = '../images/nobu/'
model_dir = "./model/nobu/"
model_name = "nobunaga_model"

# 将所有的图片resize成100*100
w = 100
h = 100
c = 3


# 读取图片
def read_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            print('reading the images:%s' % (im))
            print('classify is %d:' % (idx))
            # 打开图片
            img = Image.open(im)
            img = img.resize((w,h))
            img = np.array(img)
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)
```
这段代码主要做了几件事情：
1. 把相关的工具给引用进来
2. 设置一些全局变量，比如输入图片的路径、模型输出的路径以及规定训练用的图片大小
3. 定义一个函数read_img来读取数据集，给定一个图片根目录，会自动读取其每个子目录下的图片，并且不同子目录的图片对应的分类号不同

在读取图片时进行了一些转化，比如将原始图片大小转为100*100，然后转成了一个``(100,100,3)``的NpArray，再把这个NpArray加入倒一个List中去，作为数据集，结构就成了``[n,100,100,3]``；同理，这样构造一个标签集合labels。

接下来要对数据进行一个划分，分为训练集和验证集，先对这些图片的顺序进行shuffle（洗牌），然后按照一个比例划分。

```
data, label = read_img(img_path)

# 打乱顺序
num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]

# 将所有数据分为训练集和验证集
ratio = 0.8
s = np.int(num_example * ratio)
x_train = data[:s]
y_train = label[:s]
x_val = data[s:]
y_val = label[s:]
```
至此，载入数据的工作完成。

## 三.构建CNN模型
做图片分类的首选是卷积神经网络（CNN），当然目前有很多优秀的CNN模型可以使用，我这里参考了博文 [http://www.cnblogs.com/denny402/p/6931338.html](http://www.cnblogs.com/denny402/p/6931338.html) 给出的模型，代码编写如下：
```
# -----------------构建网络----------------------
# 占位符
x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

# 第一个卷积层（100——>50)
conv1 = tf.layers.conv2d(
    inputs=x,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# 第二个卷积层(50->25)
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# 第三个卷积层(25->12)
conv3 = tf.layers.conv2d(
    inputs=pool2,
    filters=128,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

# 第四个卷积层(12->6)
conv4 = tf.layers.conv2d(
    inputs=pool3,
    filters=128,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

re1 = tf.reshape(pool4, [-1, 6 * 6 * 128])

# 全连接层
dense1 = tf.layers.dense(inputs=re1,
                         units=1024,
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
dense2 = tf.layers.dense(inputs=dense1,
                         units=512,
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
logits = tf.layers.dense(inputs=dense2,
                         units=5,
                         activation=None,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
# ---------------------------网络结束---------------------------

loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

```
这段代码构造了一个4层卷积+池化的CNN网络，然后用全连接层进行输出，这个全连接层包括两个隐层，用``ReLU``作为激活函数，一个输出层。最后，定义了损失函数、优化器、正确率度量和ACC。
至此，CNN模型构建完毕。

## 四.训练模型，保存
首先，定义了一个函数来按批次取数据进行训练
```
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]
```

接着，定义总共进行多少轮训练，以及每轮训练使用的数据量大小batch。我使用的数据集有近400张图片，训练集80%，这里40轮，每轮64张，总共2560，相当于被张图片拿来训练了7、8次左右。然后创建了一个长连接的InteractiveSession。

```
n_epoch = 40
batch_size = 64
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
```
接下来就是训练和保存模型了。这里创建一个保存器，定义最多保存3个模型，创建模型生成路径，然后进行迭代训练，训练时打印每一轮的训练误差、ACC以及验证误差、ACC。迭代40轮之后ACC已经达到0.9以上了，试过增加迭代轮数ACC可以接近1。
```
# 保存模型
saver=tf.train.Saver(max_to_keep=3)
max_acc=0

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

for epoch in range(n_epoch):
    start_time = time.time()
    print ("step:\t%d" % epoch)
    # training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
        print x_train_a.shape
        train_loss += err;
        train_acc += ac;
        n_batch += 1
    print("   train loss: %f" % (train_loss / n_batch))
    print("   train acc: %f" % (train_acc / n_batch))

    # validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err;
        val_acc += ac;
        n_batch += 1
    print("   validation loss: %f" % (val_loss / n_batch))
    print("   validation acc: %f" % (val_acc / n_batch))

    # 保存模型
    if val_acc > max_acc:
        max_acc = val_acc
        saver.save(sess, os.path.join(model_dir, model_name), global_step = epoch+ 1)
        print "保存模型成功！"

sess.close()
```
这里有个技巧是每轮迭代计算新的验证ACC是否大于历史最有ACC，如果大于则把模型保存下来，否则就不保存。因为前面设置了最多保留3个模型，因此训练完后保留了ACC最高的3个模型。

## 五.加载模型和预测
新建一个CnnPredict的Python脚本，首先要把刚才定义的模型结构搬过来，因为代码太多，这里就省略了（如果有可以把结构也保存到模型的方法你可以告诉我，我试过不把模型搬过来Saver就会报错，说没有模型需要保存）。

接着是把sess和saver都还原出来，然后使用latest_checkpoint在生成的一堆模型中去找最后一个生成的模型，进行还原。
```
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

# Load model
model_file=tf.train.latest_checkpoint(model_dir)
saver.restore(sess, model_file)

```

接下来就要把之前被我拎出来的信长的头像拿来做预测了。这里需要构造跟之前训练一样的数据格式，不然会报格式不匹配的错误。

```
# 加载信长头像，正确的分类是0
imgs = []
labels = []

img = Image.open(img_path + "00034_00001.jpg")
img = img.resize((w, h))
img = np.array(img)
imgs.append(img)
labels.append(0)

imgs = np.asarray(imgs, np.float32)
labels = np.asarray(labels, np.float32)

ret = sess.run(y_, feed_dict={x: imgs, y_:labels})
print("计算模型结果成功！")
# 显示测试结果
print("预测结果:%d" % ret)
print("实际结果:%d" % 0)

# 加载信喵头像，正确的分类是1
imgs = []
labels = []

img = Image.open(img_path + "00034_01904.jpg")
img = img.resize((w, h))
img = np.array(img)
imgs.append(img)
labels.append(1)

imgs = np.asarray(imgs, np.float32)
labels = np.asarray(labels, np.float32)

# 根据模型计算结果
ret = sess.run(y_, feed_dict={x: imgs, y_:labels})
print("计算模型结果成功！")
# 显示测试结果
print("预测结果:%d" % ret)
print("实际结果:%d" % 1)
sess.close()
```

预测结果如下：
```
计算模型结果成功！
预测结果:0
实际结果:0
计算模型结果成功！
预测结果:1
实际结果:1
```
很高兴，我们的机器能够正确识别信长和信喵了。
![织田信长](https://raw.githubusercontent.com/LeeKrSe/TravelToNorthWest/master/blog_img/nobunaga/00034_00001.jpg)
织田信长
![织田信喵](https://raw.githubusercontent.com/LeeKrSe/TravelToNorthWest/master/blog_img/nobunaga/00034_01904.jpg)
织田信喵

## 参考文献：
[1] [tensorflow 1.0 学习：用CNN进行图像分类](http://www.cnblogs.com/denny402/p/6931338.html)        
[2] [Tensorflow模型保存与使用](http://wustmeiming.github.io/2017/01/09/Tensorflow%E6%A8%A1%E5%9E%8B%E4%BF%9D%E5%AD%98%E4%B8%8E%E4%BD%BF%E7%94%A8/)

## 完整代码
[见我的github](https://github.com/LeeKrSe/TensorFlowDemo/tree/master/nobunaga/ImageClassify)