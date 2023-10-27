import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 加载数据 进行独热编码
mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 输入像素点个数
input_size = 28 * 28
# 分类
num_class = 10


def weight_variable(shape):
    """
    构建权重w  产生随机变量
    """
    w = tf.random_normal(shape=shape, stddev=0.01)
    return tf.Variable(w)


def bias_variable(shape):
    """
    构建偏置b
    :param shape:
    :return:
    """
    b = tf.constant(0.01, shape=shape)
    return tf.Variable(b)


def con2d(x, w):
    """
    卷积层准备：函数实现
    :param x: 图像矩阵信息
    :param w: 卷积核值
    strides 卷积步长[上右下左]
    padding：
    SAME：补原来一样大小的
    VALID：缩小，不够的地方不补充
    :return: conv2d  卷积函数
    """
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def relu_con(x):
    """
    激活卷积层 x与0比较
    :param x:
    :return:
    """
    return tf.nn.relu(x)

def max_pool_con(x):
    """
    池化层 小一倍数据
    :param x:图片矩阵信息
    ksize：
    strides：步长
    padding：与卷积的padding一样
    :return:
    """
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
# 网络搭建
# 特征标签的占位符
# xs输入特征值的像素 None表示行数 input_size表示列数
xs = tf.placeholder(tf.float32, shape=[None, input_size])
# ys识别的数字，有几种类别
ys = tf.placeholder(tf.float32, shape=[None, num_class])
# 28*28的矩阵 -1任意张  1表示深度
x_image = tf.reshape(xs, [-1, 28, 28, 1])

# CNN构建
# 第一层
# 卷积层  卷积核patch=5*5 输入数据的维度--1  输出高度--32
# 为什么是1？
# 因为输入的图片是灰度处理后的所以是1
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# 卷积
conv1 = con2d(x_image,w_conv1)
# 激活
h_conv1 = relu_con(conv1+b_conv1)
# 池化
h_pool1 = max_pool_con(h_conv1)

# 第二层
# 上面的输出下面的输入
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
# 卷积
conv2 = con2d(h_pool1,w_conv2)
# 激活
h_conv2 = relu_con(conv2+b_conv2)
# 池化
h_pool2 = max_pool_con(h_conv2)

# 全连层   实现数据拍平、分类
# 全连层1：前馈神经网络
# 第二次池化后的输出
# 原input_size=28一层池化后少一半前面有两层此处就是7，前面输出64所以是7*7*64
# 假设神经元个数1024（2的次方数）
w_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
# 输出矩阵 行不管  列：7*7*64
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
# wx+b
h_fc1 = tf.matmul(h_pool2_flat,w_fc1)+b_fc1
# 激活
h_fc1 = relu_con(h_fc1)

# 全连层2：分类 输出
w_fc2 = weight_variable([1024,num_class])
b_fc2 = bias_variable([num_class])
# 计算激活
h_fc2 = tf.matmul(h_fc1,w_fc2)+b_fc2
# 激活——分类
predict = tf.nn.softmax(h_fc2)

# 构建损失函数 前面的文章有介绍
loss = tf.reduce_mean(-tf.reduce_mean(ys*tf.log(predict),reduction_indices=[1]))
# 优化器 梯度下降函数
Optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)
init = tf.global_variables_initializer()
# 训练次数
train_times = 500
# 分批训练
batch_size = 100
# 启动绘画，开始训练
with tf.Session() as sess:
    sess.run(init)
    for i in range(train_times):
        batch_x,batch_y = mnist_data.train.next_batch(batch_size)
        sess.run(Optimizer,feed_dict={xs:batch_x,ys:batch_y})
        train_loss = sess.run(loss,feed_dict={xs:batch_x,ys:batch_y})
        print(train_loss)
