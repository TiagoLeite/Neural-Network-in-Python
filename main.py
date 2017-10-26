import tensorflow as tf
import os
import sys
import datetime
import pickle
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

n_classes = 10
x = tf.placeholder('float', [1024])
y = tf.placeholder('float', [n_classes])
keep_prob = tf.placeholder(tf.float32)  # for dropout


def mix_channels(array_rgb):

    size = int(len(array_rgb)/3)
    array = [float]*size

    # print("RGB:\n", array_rgb)

    for k in range(0, int(size)):
        array[k] = array_rgb[k] + array_rgb[k+1024]*256 + array_rgb[k+1024]*256*256
        array[k] /= (np.power(2, 24)-1)

    # print(array)

    # print("max = ", max(array))

    return array


def unplicke(file):
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#                               size of window    movement of window


def convolutional_neural_network(x):

    weigths = {'w_conv1': tf.Variable(tf.random_normal([3, 3, 1, 32])),
               'w_conv2': tf.Variable(tf.random_normal([4, 4, 32, 64])),
               'w_fc': tf.Variable(tf.random_normal([8*8*64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 32, 32, 1])

    conv1 = tf.nn.relu(conv2d(x, weigths['w_conv1'])+biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weigths['w_conv2'])+biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 8*8*64])
    fc = tf.nn.relu(tf.matmul(fc, weigths['w_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_prob)

    output = tf.matmul(fc, weigths['out'])+biases['out']

    return output


def train_neural_network(x):  # x is input data

    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 0))

    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for p in range(5):

            dictionary = unplicke('data_batch_'+str(p+1))
            # print(dictionary)
            print('data_batch_'+str(p+1))
            array_data = dictionary[b'data']
            labels = dictionary[b'labels']

            print("Training...")
            #  start_time = datetime.datetime.now()
            for k in range(10000):

                batch = array_data[k]  # batch size = 3072
                batch = mix_channels(batch)
                # print(batch)
                # print(len(batch))
                label = [0] * 10
                label[labels[k]] = 1
                # print(label)

                sess.run([optimizer, cost], feed_dict={x: batch, y: label, keep_prob: 0.9})
                # print(sess.run([prediction], feed_dict={x: batch, y: label, keep_prob: 0.9}))
                progress(k, 10000)

                '''if k % 100 == 0:
                    train_acc = accuracy.eval(feed_dict={x: batch, y: label, keep_prob: 1.0})
                    print('\nReached step %d with training accuracy %g\n' % (k, train_acc))'''

            #  time_end = datetime.datetime.now()
            #  print("\n\nFinished training in", (time_end - start_time))
            #  print("\nTesting...")
        dictionary = unplicke('test_batch')
        array_test = dictionary[b'data']
        labels_test = dictionary[b'labels']
        acc = 0
        for k in range(10000):
            batch = array_test[k]
            batch = mix_channels(batch)
            label = [0] * 10
            label[labels_test[k]] = 1
            # sess.run([optimizer, cost], feed_dict={x: batch, y: label, keep_prob: 1})
            acc += accuracy.eval(feed_dict={x: batch, y: label, keep_prob: 1.0})
            # print(acc, "/", (k+1))
        print("\nTesting Accuracy: ", acc/10000)


def progress(prog, total):  # to show progress bar
    if prog <= 0:
        return
    sys.stdout.write('\r')
    sys.stdout.write("[%-100s] %d%%" % (u"\u2588"*int(100*((prog+1)/total)), 100*(prog+1)/total))
    sys.stdout.flush()


train_neural_network(x)
























