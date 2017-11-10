# Tiago de Miranda Leite, 7595289
# Atenção: os datasets do cifar-10 nao foram inclusos para nao tornar os arquivos de submissão muito extensos

import tensorflow as tf
import os
import datetime
import pickle
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

n_classes = 10
x = tf.placeholder('float', [None, 3*1024])
y = tf.placeholder('float', [None, n_classes])
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
    for i in range(start, start+batch_size):
        batch[0].append(array_data[i])
        label_array = np.zeros(n_classes)
        label_array[labels[i]] = 1  # puts the value 1 in the corresponding position
        batch[1].append(label_array)
    return batch


def convolutional_neural_network(x):

    weigths = {'w_conv1': tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=0.1)),
               'w_conv2': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1)),
               'w_conv3': tf.Variable(tf.truncated_normal([4, 4, 128, 256], stddev=0.1)),
               'w_fc': tf.Variable(tf.truncated_normal([4*4*256, 1024], stddev=0.1)),
               'out': tf.Variable(tf.truncated_normal([1024, n_classes], stddev=0.1))}

    biases = {'b_conv1': tf.Variable(tf.constant(0.1, shape=[64])),
              'b_conv2': tf.Variable(tf.constant(0.1, shape=[128])),
              'b_conv3': tf.Variable(tf.constant(0.1, shape=[256])),
              'b_fc': tf.Variable(tf.constant(0.1, shape=[1024])),
              'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))}

    x = tf.reshape(x, shape=[-1, 32, 32, 3])

    # convolutional layer 1:
    conv1 = tf.nn.relu(conv2d(x, weigths['w_conv1'])+biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    # convolutional layer 2:
    conv2 = tf.nn.relu(conv2d(conv1, weigths['w_conv2'])+biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    # convolutional layer 3:
    conv3 = tf.nn.relu(conv2d(conv2, weigths['w_conv3']) + biases['b_conv3'])
    conv3 = maxpool2d(conv3)

    # fully connected layer
    fc = tf.reshape(conv3, [-1, 4*4*256])
    fc = tf.nn.relu(tf.matmul(fc, weigths['w_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_prob)

    # output layer
    output = tf.matmul(fc, weigths['out'])+biases['out']

    return output


def train_neural_network(x):  # x is the input data

    epochs = 3
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_test = get_batch(0, 1000, 5)  # 1000 images for testing while training so we can see the evolution of accuracy
        start_time = datetime.datetime.now()
        print("Training...")
        for epoch in range(epochs):
            print("Started epoch: ", (epoch+1), '/', epochs)
            for file_train in range(5):  # there are 5 files for training
                batches = 200
                for k in range(batches):
                    batch = get_batch(50 * k, 50, file_train)  # gets the next 50 train images
                    sess.run(optimizer, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.9})
                    if k % 10 == 0:
                        print('Reached step %3d' % k, '(of 200) of train file', (file_train+1), '(of 5) with accuracy ', end='')
                        print(accuracy.eval(feed_dict={x: batch_test[0], y: batch_test[1], keep_prob: 1.0}))
        time_end = datetime.datetime.now()
        print("\nFinished training in", (time_end - start_time))
        print("Epochs: ", epochs)
        print("Testing...")
        file_test = 5  # the 5th element corresponds to the test file in the datasets list
        batch_test = get_batch(0, 10000, file_test)  # loads the whole test dataset
        print('Testing accuracy =', end=' ')
        print(accuracy.eval(feed_dict={x: batch_test[0], y: batch_test[1], keep_prob: 1.0}))


load_datasets()
train_neural_network(x)
