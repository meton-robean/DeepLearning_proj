
#cmt 2018-04-29
#simple test for using tensorflow Saver class to store training process checkpoint~
#blog: https://blog.csdn.net/u011500062/article/details/51728830


import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, shape=[None, 1])
y = 4 * x + 4

w = tf.Variable(tf.random_normal([1], -1, 1))
b = tf.Variable(tf.zeros([1]))
y_predict = w * x + b


loss = tf.reduce_mean(tf.square(y - y_predict))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# isTrain：用来区分训练阶段和测试阶段，True表示训练，False表示测试
# train_steps：表示训练的次数，例子中使用100
# checkpoint_steps：表示训练多少次保存一下checkpoints，例子中使用50
# checkpoint_dir：表示checkpoints文件的保存路径，例子中使用当前路
isTrain = False
train_steps = 100
checkpoint_steps = 50
checkpoint_dir = './save_checkpoint/'

saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
x_data = np.reshape(np.random.rand(10).astype(np.float32), (10, 1))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    if isTrain:
        for i in range(train_steps):
            sess.run(train, feed_dict={x: x_data})
            if (i + 1) % checkpoint_steps == 0:
                saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=i+1)
    else:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            pass
        print(sess.run(w))
        print(sess.run(b))

