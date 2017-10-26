import tensorflow as tf
import os
import sys
import datetime
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

n_classes = 10
batch_size = 50
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])
keep_prob = tf.placeholder(tf.float32)  # for dropout


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#                               size of window    movement of window


def convolutional_neural_network(x):

    weigths = {'w_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'w_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'w_fc': tf.Variable(tf.random_normal([7*7*64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(x, weigths['w_conv1'])+biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weigths['w_conv2'])+biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weigths['w_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_prob)

    output = tf.matmul(fc, weigths['out'])+biases['out']

    return output


def train_neural_network(x):  # x is input data

    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print("Training...")
        start_time = datetime.datetime.now()

        for k in range(1000):
            batch = mnist.train.next_batch(batch_size)
            sess.run([optimizer, cost], feed_dict={x: batch[0], y: batch[1], keep_prob: 0.8})
            progress(k, 1000)
            if k % 100 == 0:
                train_acc = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
                print('\nReached step %d with training accuracy %g\n' % (k, train_acc))

        time_end = datetime.datetime.now()

        print("\n\nFinished training in", (time_end - start_time))
        print("\nTesting...")
        print("Testing Accuracy: ", accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))


def progress(prog, total):  # to show progress bar
    if prog <= 0:
        return
    sys.stdout.write('\r')
    sys.stdout.write("[%-100s] %d%%" % (u"\u2588"*int(100*((prog+1)/total)), 100*(prog+1)/total))
    sys.stdout.flush()


train_neural_network(x)
























