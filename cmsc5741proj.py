import numpy as np
import tensorflow as tf

server_target = "grpc://localhost:12345"

train = []
with open('./text_vector/text_vector.txt') as f:
    for line in f:
        txt = line.split(" ")[1:-1]
        txt = list(map(float, txt))
        train.append(txt)
train = np.array(train)
M = train.shape[1]

label = []
with open('./text_vector/text_label.txt') as f:
    for line in f:
        txt = int(line.split(" ")[1])
        txt = -1 if txt == 0 else 1
        label.append(txt)
label = np.array(label)
label = label.reshape(label.shape[0], 1)

with tf.device("/job:worker/task:0"):
    x_data = tf.placeholder(shape=[None, M], dtype=tf.float32)
    x_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    A = tf.Variable(tf.random_normal(shape=[M, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))

with tf.device("/job:ps/task:0"):
    model_output = tf.subtract(tf.matmul(x_data, A), b)
    l2_norm = tf.reduce_sum(tf.square(A))
    alpha = tf.constant([0.01])

    classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, x_target))))
    loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session(server_target) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(2000):
        sess.run(train_step, feed_dict={x_data: train, x_target: label})
        if i % 100 == 0:
            print(sess.run(loss, feed_dict={x_data: train, x_target: label}))

saver = tf.train.Saver()
saver.save(sess, "./MoModel")
