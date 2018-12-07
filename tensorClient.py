import tensorflow as tf
import numpy as np

server_target = "grpc://localhost:12345"

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

with tf.device("/job:worker/task:0"):
    Weight = tf.Variable(tf.random_uniform([1],-1.0,1.0))
    biases = tf.Variable(tf.zeros([1]))

with tf.device("/job:ps/task:0"):
    y = Weight * x_data + biases
    loss = tf.reduce_mean(tf.square(y-y_data))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    init = tf.global_variables_initializer()

with tf.Session(server_target) as sess:
    sess.run(init)

    for step in range(1000):
        sess.run(train_step)
        if step % 50 == 0:
            print(sess.run(Weight),' ', sess.run(biases))