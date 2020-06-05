import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt#可视化
# 定义一个神经层
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))#随机变量normal
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)#1行，out_size列，初始值不为0，加0.1
    Wx_plus_b = tf.matmul(inputs,Weights)+biases#预测值没有被激活之前被存储在这个变量里,matmul是矩阵相乘

    #激活
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#构造训练集数据,300个样本点，特征维度1维
x_data = np.linspace(-1,1,300)[:,np.newaxis]#x_data是-1到1之间的300个数，[]中的意思是加一个维度，，变为矩阵，300行 1列
noise = np.random.normal(0,0.05,x_data.shape)#生成一些随机噪声，更像真实数据。平均值为0，标准差是0.05，和x_data一样的格式
y_data= np.square(x_data)-0.5 + noise#非线性函数，x的平方赋值给y，最后减去0.5

xs = tf.placeholder(tf.float32,[None,1])#x_data的结构，行不定，列为1
ys = tf.placeholder(tf.float32,[None,1])

#神经网络解释：输入只有一个属性（一列），就是一个神经元，输出也是一个属性（一列），即一个神经元。现在隐藏层设定10个神经元

layer1 = add_layer(xs,1,10,activation_function=tf.nn.sigmoid)#隐藏层1输入为xs,输入为1列，输出为10列，layer1就相当于隐藏层的输出
prediction = add_layer(layer1,10,1,activation_function=None)#输出层定义，输入数据为layer1，即10列，输出为1即结果1列


loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                                    reduction_indices=[1]))#损失函数，计算误差，对于每一个差值计算平方square，再求和reduce_sum算出所有值,再平均
                                                            #reduction_indices=[1]表示按行求和，0表示按列求和
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)#学习率0.1
                                                                    #每一步训练都使用优化器来对loss进行更正
init = tf.initialize_all_variables()
# with tf.Session() as sess:
sess = tf.Session()
sess.run(init)
#可视化操作
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()
for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})#使用xs和ys的意义是不把程序写死，xs和ys作形参
    if i%50 == 0:
        # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        # try:
        #     ax.lines.remove(lines[0])#抹除掉lines的第一个单位
        # except Exception:
        #     pass
        # prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        # lines = ax.plot(x_data,prediction_value,'r-',lw=5)#用宽度为5红色的线把根据x_data预测的值画出来
        # #如果想在if语句中每循环一次就画一次线，就需要抹除掉
        # plt.pause(0.1)#暂停0.1秒继续显示
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        lines = ax.plot(x_data,prediction_value,'r-',lw=5)#用宽度为5红色的线把根据x_data预测的值画出来




