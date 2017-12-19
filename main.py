# Tiago de Miranda Leite, 7595289

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
C1, C2, C3 = 30, 50, 80
F1 = 500


def unplicke(file):
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


def conv2d(input_data, W):
    return tf.nn.conv2d(input_data, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(input_data):
    return tf.nn.max_pool(input_data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
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


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv_layer(input_data, shape):
    W = weight_variable(shape=shape)
    b = bias_variable(shape=[shape[3]])
    return tf.nn.relu(conv2d(input_data, W) + b)


def full_layer(input_data, out_size):
    in_size = int(input_data.get_shape()[1])
    W = weight_variable([in_size, out_size])
    b = bias_variable([out_size])
    return tf.matmul(input_data, W) + b


def new_model(x):
    x_image = tf.reshape(x, [-1, 32, 32, 3])
    conv1 = conv_layer(x_image, shape=[5, 5, 3, 32])
    conv1_pool = maxpool2d(conv1)
    conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
    conv2_pool = maxpool2d(conv2)
    conv2_flat = tf.reshape(conv2_pool, shape=[-1, 8 * 8 * 64])
    full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))
    full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)
    y_conv = full_layer(full1_drop, 10)
    return y_conv


def train_neural_network(x):  # x is the input data
    epochs = 20
    # prediction = convolutional_neural_network(x)
    prediction = new_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(cost)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_test = get_batch(0, 1000, 5)  # 500 images for testing while training so we can see the evolution of accuracy
        start_time = datetime.datetime.now()
        print("Training...")
        for epoch in range(epochs):
            print("Started epoch: ", (epoch+1), '/', epochs)
            for file_train in range(5):  # there are 5 files for training
                batches = 100
                for k in range(batches):
                    batch = get_batch(100 * k, 100, file_train)  # gets the next 50 train images
                    sess.run(optimizer, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
                    if k % 10 == 0:
                        print('Reached step %3d' % k, '(of 100) of train file', (file_train+1), '(of 5) with accuracy ', end='')
                        print(accuracy.eval(feed_dict={x: batch_test[0], y: batch_test[1], keep_prob: 1.0}))
            save_path = saver.save(sess, "save/saved_net.ckpt")
            print("Saved to:", save_path)
        time_end = datetime.datetime.now()
        print("\nFinished training in", (time_end - start_time))
        print("Epochs: ", epochs)
        print("Testing...")
        file_test = 5  # the 5th element corresponds to the test file in the datasets list
        for k in range(10):
            batch_test = get_batch(k*1000, 1000, file_test)
            print("Size test:", len(batch_test[0]))
            print('Test %d accuracy =' % k, end=' ')
            print(accuracy.eval(feed_dict={x: batch_test[0], y: batch_test[1], keep_prob: 1.0}))
            print("loss:", cost.eval(feed_dict={x: batch_test[0], y: batch_test[1], keep_prob: 1.0}))


load_datasets()
train_neural_network(x)
