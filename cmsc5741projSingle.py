import numpy as np
import tensorflow as tf

server_target = "grpc://localhost:12345"

train = []
<<<<<<< HEAD
with open('./data/test_text_vector.txt') as f:
=======
with open('./text_vector/text_vector.txt') as f:
>>>>>>> 16d4910dd50d6f60448c6ac9cc37391f026788c9
    for line in f:
        txt = line.split(" ")[1:-1]
        txt = list(map(float, txt))
        train.append(txt)
train = np.array(train)
<<<<<<< HEAD

# train = PCA(n_components=0.9).fit_transform(train)

M = train.shape[1]

label = []
with open('./data/test_text_label.txt') as f:
=======
M = train.shape[1]

label = []
with open('./text_vector/text_lable.txt') as f:
>>>>>>> 16d4910dd50d6f60448c6ac9cc37391f026788c9
    for line in f:
        txt = int(line.split(" ")[1])
        txt = -1 if txt == 0 else 1
        label.append(txt)
label = np.array(label)
label = label.reshape(label.shape[0], 1)

x_data = tf.placeholder(shape=[None, M], dtype=tf.float32)
x_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

A = tf.Variable(tf.random_normal(shape=[M, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

<<<<<<< HEAD
# A = tf.Variable(tf.random_normal(shape=[1, train.shape[0]]))


# not kernel function
model_output = tf.subtract(tf.matmul(x_data, A), b)

# compute gaussian kernel
# gamma = tf.constant(-10.0)
# dist = tf.reduce_sum(tf.square(x_data), 1)
# dist = tf.reshape(dist, [-1, 1])
# sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))), tf.transpose(dist))
# kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))
# model_output = tf.subtract(tf.matmul(A, kernel), b)

=======
model_output = tf.subtract(tf.matmul(x_data, A), b)
>>>>>>> 16d4910dd50d6f60448c6ac9cc37391f026788c9
l2_norm = tf.reduce_sum(tf.square(A))
alpha = tf.constant([0.01])

classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, x_target))))
loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

<<<<<<< HEAD
isTrain = False

if isTrain:
    for i in range(3000):
        sess.run(train_step, feed_dict={x_data: train, x_target: label})
        if i % 100 == 0:
            print(sess.run(loss, feed_dict={x_data: train, x_target: label}))

    saver = tf.train.Saver()
    saver.save(sess, "./model/MoModel_test")
else:
    saver = tf.train.Saver()
    saver.restore(sess, './model/MoModel')

    result = sess.run(model_output, feed_dict={x_data: train})


    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(result)):
        if result[i] > 0 and label[i] > 0:
            TP += 1
        elif result[i] < 0 and label[i] > 0:
            FN += 1
        elif result[i] > 0 and label[i] < 0:
            FP += 1
        else:
            TN += 1

    p_distribute = TP + FP
    m_distribute = TN + FN

    precision = float(TP) / (TP + FP)
    accuracy = float(TP + TN) / (TP + FP + TN + FN)
    recall = float(TP) / (TP + FN)
    f1 = precision * recall / (precision + recall)

    print("precision is %f" % precision)
    print("accuracy is %f" % accuracy)
    print("recall is %f" % recall)
    print("f1 score is %f" % f1)
=======
for i in range(2000):
    sess.run(train_step, feed_dict={x_data: train, x_target: label})
    if i % 100 == 0:
        print(sess.run(loss, feed_dict={x_data: train, x_target: label}))

saver = tf.train.Saver()
saver.save(sess, "./MoModel")
>>>>>>> 16d4910dd50d6f60448c6ac9cc37391f026788c9
