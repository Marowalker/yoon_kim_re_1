import tensorflow as tf

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

init = tf.initialize_all_variables()

#placeholder la cac tham so ma se duoc dien trong qua trinh training
#shape [None, 28, 28, 1] nghia la anh co 28x28x1 (gray scale 1)
#None la so mini-batch
#Variable la cac tham so ma muon thuat toan xac dinh cho minh

#model
Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W) + b)
Y_ = tf.placeholder(tf.float32, [None, 10]) # placeholder chua label dung

cross_entropy = -tf.reduce_sum(Y_*log(Y))

is_correct = tf.equal(tf.arg_max(Y, 1), tf.arg_max(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct,  tf.float32))
optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

sess = tf.Session()
sess.run(init)

#for i in range(1000):
#    batch_X, batch_Y  = mnist.train.next_batch(100)
#    train_data = {X: batch_X, Y: batch_Y}
#    sess.run(train_step, feed_dict=train_data)

#a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)

#test_data = {X: mnist.test.images, Y_: mnist.test.labels}
#a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)

W1 = tf.Variable(tf.truncated_normal([28*28, 200], stddev = 0.1))
B1 = tf.Variable(tf.zeros[200])

W2 = tf.Variable(tf.truncated_normal([200, 10], stddev = 0.1))
B1 = tf.Variable(tf.zeros[10])


W2 = tf.Variable(tf.truncated_normal([200, 100], stddev = 0.1))
B2 = tf.Variable(tf.zeros[100])

XX = tf.reshape(X, [-1, 28*28])
Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1)
Y = tf.nn.softmax(tf.matmul(Y1, W2) + B2)

