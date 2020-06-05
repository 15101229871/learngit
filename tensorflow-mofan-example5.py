import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)

# init = tf.initialize_all_variables()

with tf.Session() as sess:
    # sess.run(init)
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))#placeholder相当于一个占位符，在输出的时候用到时再给定值。
                                                             #feed_dict是一个字典形式。[7.]就是[7.0]