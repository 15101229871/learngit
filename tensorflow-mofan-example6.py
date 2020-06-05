import tensorflow as tf

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))#随机变量normal
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)#1行，out_size列，初始值不为0，加0.1
    Wx_plus_b = tf.multiply(inputs,Weights)+biases#预测值没有被激活之前被存储在这个变量里

    #激活
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
