import tensorflow as tf  # 导入TensorFlow的包
import numpy as np

#create data
x_data = np.random.rand(100).astype(np.float32)#随机生成100个32位浮点型数据
y_data = x_data*0.1 + 0.3#要使得weights接近0.1，biases接近0.3，也就是学习训练过程


###create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))#考虑到权重可能是一个矩阵的形式，所以大写W。
                                                      #定义参数要用到Variable，生成一个随机数，[1]表示一维，从-1到1
biases = tf.Variable(tf.zeros([1]))#偏置初始值为0，[1]为一维，训练的过程就是不断从初始值去接近0.1和0.3

y = Weights*x_data + biases #定义要预测的y的值

loss = tf.reduce_mean(tf.square(y-y_data))#计算预测的y和真实y的差距，刚开始loss会很大
optimizer = tf.train.GradientDescentOptimizer(0.5)#建立一个优化器来减小误差loss，提升参数的准确度，选择最基础的原始的梯度下降优化器，学习率0.5（学习率要小于1）
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()#初始化所有变量
###create tensorflow structure end ###

#激活神经网络
sess = tf.Session()
sess.run(init)#也可以用with tf.Sess() as sess:

#开始训练
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(biases))#每隔20次输出一次结果，在输出weights和biases时注意要sess.run()
