import tensorflow as tf
import os
import sys
import datetime
import pickle
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

n_classes = 10
x = tf.placeholder('float', [None, 1024])
y = tf.placeholder('float', [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # for dropout
datasets = []


def resize_array(array_rgb):
    size = int(len(array_rgb)/3)
    array = [float]*size
    max_value = (np.power(2, 24)-1)
    for k in range(0, int(size)):
        array[k] = array_rgb[k] + array_rgb[k+1024]*256 + array_rgb[k+1024*2]*256*256
        array[k] /= max_value
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


def load_datasets():
    for k in range(5):
        file_name = 'data_batch_' + str(k + 1)
        datasets.append(unplicke(file_name))
    file_name = 'test_batch'
    datasets.append(unplicke(file_name))


def get_batch(start, batch_size, dataset_index):
    dictionary = datasets[dataset_index]
    array_data = dictionary[b'data']
    labels = dictionary[b'labels']
    batch = [[], []]
    for i in range(start, start+batch_size):
        batch[0].append(resize_array(array_data[i]))
        label_array = np.zeros(n_classes)
        label_array[labels[i]] = 1
        batch[1].append(label_array)
    return batch


def convolutional_neural_network(x):

    weigths = {'w_conv1': tf.Variable(tf.random_normal([3, 3, 1, 32])),
               'w_conv2': tf.Variable(tf.random_normal([3, 3, 32, 64])),
               'w_conv3': tf.Variable(tf.random_normal([3, 3, 64, 128])),
               'w_fc': tf.Variable(tf.random_normal([4*4*128, 1024])),
               'w_fc2': tf.Variable(tf.random_normal([1024, 256])),
               'out': tf.Variable(tf.random_normal([256, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_conv3': tf.Variable(tf.random_normal([128])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'b_fc2': tf.Variable(tf.random_normal([256])),
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

    fc2 = tf.nn.relu(tf.matmul(fc, weigths['w_fc2']) + biases['b_fc2'])
    fc2 = tf.nn.dropout(fc2, keep_prob)

    output = tf.matmul(fc2, weigths['out'])+biases['out']

    return output


def train_neural_network(x):  # x is the input data

    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        r1 = 5
        batch_test = get_batch(0, 500, 5)
        start_time = datetime.datetime.now()
        print("Training...")
        for j in range(5):
            print("Epoch: ", j)
            for p in range(r1):
                r = 200
                for k in range(r):
                    batch = get_batch(50 * k, 50, p)
                    sess.run(optimizer, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.9})
                    if k % 10 == 0:
                        print("Partial test accuracy:", k, end=' ')
                        print(accuracy.eval(feed_dict={x: batch_test[0], y: batch_test[1], keep_prob: 1.0}))
        time_end = datetime.datetime.now()
        print("\n\nFinished training in", (time_end - start_time))
        file_name = 'test_batch'
        print(file_name)
        print("\tTesting...")
        batch_test = get_batch(0, 10000, 5)
        print('Testing accuracy =', end=' ')
        print(accuracy.eval(feed_dict={x: batch_test[0], y: batch_test[1], keep_prob: 1.0}))


def progress(prog, total):  # to show progress bar
    if prog <= 0:
        return
    sys.stdout.write('\r')
    sys.stdout.write("%-100s] %d%%" % (u"\u2588"*int(100*((prog+1)/total)), 100*(prog+1)/total))
    sys.stdout.flush()


load_datasets()
train_neural_network(x)
