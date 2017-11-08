import tensorflow as tf
import os
import sys
import datetime
import random
import pickle
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

n_classes = 10
x = tf.placeholder('float', [None, 1024])
y = tf.placeholder('float', [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # for dropout

glob_labels = [0]*10


def mix_channels(array_rgb):
    size = int(len(array_rgb)/3)
    array = [float]*size
    # print("RGB:\n", array_rgb)
    for k in range(0, int(size)):
        array[k] = array_rgb[k] + array_rgb[k+1024]*256 + array_rgb[k+1024*2]*256*256
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


def get_next_batch(start, batch_size, file_name):
    dictionary = unplicke(file_name)
    array_data = dictionary[b'data']
    labels = dictionary[b'labels']
    batch = [[], []]
    for i in range(start, start+batch_size):
        batch[0].append(mix_channels(array_data[i]))
        # print(len(array_data[i]))
        label_array = np.zeros(n_classes)
        label_array[labels[i]] = 1
        batch[1].append(label_array)
        glob_labels[labels[i]] += 1;
        # print(label_array)
        # print(array_data[i])
    # print("0:", len(batch[0]))
    return batch


def convolutional_neural_network(x):

    weigths = {'w_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'w_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'w_conv3': tf.Variable(tf.random_normal([5, 5, 64, 128])),
               'w_fc': tf.Variable(tf.random_normal([4*4*128, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_conv3': tf.Variable(tf.random_normal([128])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 32, 32, 1])

    conv1 = tf.nn.relu(conv2d(x, weigths['w_conv1'])+biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weigths['w_conv2'])+biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    conv3 = tf.nn.relu(conv2d(conv2, weigths['w_conv3']) + biases['b_conv3'])
    conv3 = maxpool2d(conv3)

    fc = tf.reshape(conv3, [-1, 4*4*128])
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
        r1 = 5
        start_time = datetime.datetime.now()
        for j in range(1):
            print("Epoch: ", j)
            for p in range(r1):
                # print(dictionary)
                file_name = 'data_batch_'+str(p+1)
                print(file_name)
                print("\tTraining...")
                #  start_time = datetime.datetime.now()
                r = 10000
                for k in range(1):
                    batch = get_next_batch(k * r, r, file_name)
                    print(glob_labels)
                    # print(len(batch[0][0]))
                    # print(len(batch[1][0]))
                    sess.run(optimizer, feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
                    print(cost.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0}))
                    # print(sess.run([prediction], feed_dict={x: batch, y: label, keep_prob: 0.9}))
                    # progress(cont, r*r1)
                    # cont += 1
                    # if k % 2 == 0:
                    train_acc = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
                    print('Reached step %d in epoch %d with training accuracy %.3f' % (k, j,  train_acc))

        time_end = datetime.datetime.now()
        print("\n\nFinished training in", (time_end - start_time))
        print("\nTesting...")

        file_name = 'test_batch'
        print(file_name)
        print("\tTesting...")
        #  start_time = datetime.datetime.now()
        batch = get_next_batch(0, 10000, "test_batch")
        print(len(batch[0]))
        print(accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0}))

        '''dictionary = unplicke('test_batch')
        array_test = dictionary[b'data']
        labels_test = dictionary[b'labels']
        acc = []
        r = 10000
        print("\nTesting...")
        for k in range(r):
            batch = array_test[k]
            batch = mix_channels(batch)
            label = [0.0] * 10
            label[labels_test[k]] = 1.0
            # sess.run([optimizer, cost], feed_dict={x: batch, y: label, keep_prob: 1})
            # print(correct.eval(feed_dict={x: batch, y: label, keep_prob: 1.0}))
            # print(accuracy.eval(feed_dict={x: batch, y: label, keep_prob: 1.0}))
            acc.append(accuracy.eval(feed_dict={x: batch, y: label, keep_prob: 1.0}))
            # progress(k, r)
        # print("\n", acc)
        mean = tf.reduce_mean(acc)

        end = datetime.datetime.now()

        print("\nTesting Accuracy: ", mean.eval())

        print("\nTime: ", (end-start))'''


def progress(prog, total):  # to show progress bar
    if prog <= 0:
        return
    sys.stdout.write('\r')
    sys.stdout.write("%-100s] %d%%" % (u"\u2588"*int(100*((prog+1)/total)), 100*(prog+1)/total))
    sys.stdout.flush()


train_neural_network(x)
























