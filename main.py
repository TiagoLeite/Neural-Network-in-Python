# Tiago de Miranda Leite, 7595289

import tensorflow as tf
import os
import datetime
import pickle
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

n_classes = 10
x = tf.placeholder('float', [None, 3 * 1024])
y = tf.placeholder('float', [None, n_classes])
phase = tf.placeholder(tf.bool, name='phase')
keep_prob = tf.placeholder(tf.float32)
datasets = []


def unplicke(file):
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


#                               size of window    movement of window


def load_datasets():  # loads all the train and test files to "datasets" list
    print("Loading files...")
    for k in range(5):
        file_name = 'data_batch_' + str(k + 1)
        datasets.append(unplicke(file_name))
    file_name = 'test_batch'
    datasets.append(unplicke(file_name))


def get_batch(start, batch_size, dataset_index):
    dictionary = datasets[dataset_index]
    array_data = dictionary[b'data']
    labels = dictionary[b'labels']
    batch = [[], []]  # first for the data array and second for the labels, like mnist
    for i in range(start, start + batch_size):
        batch[0].append(array_data[i])
        label_array = np.zeros(n_classes)
        label_array[labels[i]] = 1  # puts the value 1 in the corresponding position
        batch[1].append(label_array)
    return batch


def convolutional_neural_network(x):
    weights = {'w_conv1': tf.Variable(tf.truncated_normal([3, 3, 3, 32], stddev=0.05)),
               # 'w_conv1_2': tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.05)),
               'w_conv2': tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.05)),
               # 'w_conv2_2': tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.05)),
               # 'w_conv3': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.05)),
               # 'w_conv3_2': tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev=0.05)),
               'w_fc': tf.Variable(tf.truncated_normal([8 * 8 * 64, 1024], stddev=0.05)),
               'w_fc2': tf.Variable(tf.truncated_normal([1024, 512], stddev=0.05)),
               'out': tf.Variable(tf.truncated_normal([512, n_classes], stddev=0.05))}

    biases = {'b_conv1': tf.Variable(tf.constant(0.05, dtype='float32')),
              'b_conv1_2': tf.Variable(tf.constant(0.05, dtype='float32')),
              'b_conv2': tf.Variable(tf.constant(0.05, dtype='float32')),
              'b_conv2_2': tf.Variable(tf.constant(0.05, dtype='float32')),
              'b_conv3': tf.Variable(tf.constant(0.05, dtype='float32')),
              'b_conv3_2': tf.Variable(tf.constant(0.05, dtype='float32')),
              'b_fc': tf.Variable(tf.constant(0.05, dtype='float32')),
              'b_fc2': tf.Variable(tf.constant(0.05, dtype='float32')),
              'out': tf.Variable(tf.constant(0.05, dtype='float32'))}

    x = tf.reshape(x, shape=[-1, 32, 32, 3])
    x_norm = tf.nn.batch_normalization(x,
                                       0,
                                       1,
                                       offset=5,
                                       scale=True,
                                       variance_epsilon=1e-8)

    # convolutional layer 1:
    conv1 = tf.nn.elu(conv2d(x_norm, weights['w_conv1']) + biases['b_conv1'])
    norm1 = tf.nn.lrn(conv1, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm2')
    conv1_pool = maxpool2d(norm1)
    # conv1_1 = tf.nn.relu(conv2d(norm1, weights['w_conv1_2']) + biases['b_conv1_2'])
    # norm1_1 = tf.nn.lrn(conv1_1, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm2')
    # conv1_1_pool = maxpool2d(norm1_1)
    # convolutional layer 2:
    conv2 = tf.nn.elu(conv2d(conv1_pool, weights['w_conv2']) + biases['b_conv2'])
    norm2 = tf.nn.lrn(conv2, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm2')
    conv2_pool = maxpool2d(norm2)
    # conv2_2 = tf.nn.relu(conv2d(norm2, weights['w_conv2_2']) + biases['b_conv2_2'])
    # norm2_2 = tf.nn.lrn(conv2_2, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm2')
    # conv2_2_pool = maxpool2d(norm2_2)
    # convolutional layer 3:
    # conv3 = tf.nn.relu(conv2d(norm2, weights['w_conv3']) + biases['b_conv3'])
    # conv3_2 = tf.nn.relu(conv2d(conv3, weights['w_conv3_2']) + biases['b_conv3_2'])
    # conv3_2_pool = maxpool2d(conv3_2)
    # norm3 = tf.nn.lrn(conv3_2_pool, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm2')
    # fully connected layer
    fc = tf.reshape(conv2_pool, [-1, 8 * 8 * 64])
    fc_out = tf.nn.elu(tf.matmul(fc, weights['w_fc']) + biases['b_fc'])
    fc_drop = tf.nn.dropout(fc_out, keep_prob)
    # fully connected layer 2
    fc2 = tf.nn.elu(tf.matmul(fc_drop, weights['w_fc2']) + biases['b_fc2'])
    fc2_drop = tf.nn.dropout(fc2, keep_prob)
    # output layer
    output = tf.matmul(fc2_drop, weights['out']) + biases['out']

    return output


def train_neural_network(x):  # x is the input data

    epochs = 60
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_test = get_batch(0, 1000,
                               5)  # 1000 images for testing while training so we can see the evolution of accuracy
        start_time = datetime.datetime.now()
        print("Training...")
        for epoch in range(epochs):
            print("Started epoch: ", (epoch + 1), '/', epochs)
            for file_train in range(5):  # there are 5 files for training
                batches = 100
                for k in range(batches):
                    batch = get_batch(100 * k, 100, file_train)  # gets the next 50 train images
                    sess.run(optimizer, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5, phase: True})
                    if k % 10 == 0:
                        print('Reached step %3d' % k, '(of %d)' % batches, 'of train file', (file_train + 1),
                              '(of 5) with accuracy ', end='')
                        print(
                            accuracy.eval(feed_dict={x: batch_test[0], y: batch_test[1], keep_prob: 1.0, phase: False}))
        time_end = datetime.datetime.now()
        print("\nFinished training in", (time_end - start_time))
        print("Epochs: ", epochs)
        print("Testing...")
        file_test = 5  # the 5th element corresponds to the test file in the datasets list
        for k in range(10):
            batch_test = get_batch(k * 1000, 1000, file_test)
            print("Size test:", len(batch_test[0]))
            print('Test %d accuracy =' % k, end=' ')
            print(accuracy.eval(feed_dict={x: batch_test[0], y: batch_test[1], keep_prob: 1.0, phase: True}))
            print("loss:", cost.eval(feed_dict={x: batch_test[0], y: batch_test[1], keep_prob: 1.0, phase: True}))


load_datasets()
train_neural_network(x)
