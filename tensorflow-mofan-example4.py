import tensorflow as tf

state = tf.Variable(0,name='counter')#定义一个变量，值为0，名为counter
# print(state.name)#输出结果：counter:0
one = tf.constant(1)#设置常量值为1

new_value = tf.add(state,one)#变量state+常量one赋值给新变量new_value
update = tf.assign(state,new_value)#更新state的值（将new_value赋值给state）

init = tf.initialize_all_variables()#初始化所有变量,如果定义了变量，一定要写这一句

#使用with语句session会话，不需要sess.close()
with tf.Session() as sess:
    sess.run(init)#在session中一定要run一次init
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
