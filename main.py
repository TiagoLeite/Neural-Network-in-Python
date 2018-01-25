# Tiago de Miranda Leite, 7595289

import tensorflow as tf
import os
import datetime
import pickle
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

n_classes = 10
x = tf.placeholder('float', [None, 3*1024])
y_ = tf.placeholder('float', [None, n_classes])
is_training = tf.placeholder(tf.bool)

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


def batch_norm_wrapper(inputs, cond, axis, decay=0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    def f1():
        batch_mean, batch_var = tf.nn.moments(inputs, axis)
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                                             batch_mean, batch_var, beta, scale, 1e-6)

    def f2():
        return tf.nn.batch_normalization(inputs,
                                         pop_mean, pop_var, beta, scale, 1e-6)

    return tf.cond(cond, f1, f2)


def new_model(x):
    # x = tf.placeholder(tf.float32, shape=[None, 784])
    # y_ = tf.placeholder(tf.float32, shape=[None, n_classes])
    # === Model ===
    x_input = tf.reshape(x, [-1, 32, 32, 3])
    # tf.image.per_image_standardization()
    # Convolutional Layer 1:
    map_size_1 = 32
    w_conv1 = weight_variable([6, 6, 3, map_size_1])
    b_conv1 = bias_variable([map_size_1])
    # h_conv1 = tf.nn.relu(conv2d(x_input, w_conv1) + b_conv1)
    # h_pool1 = max_pool_2x2(h_conv1)
    # logits_1 = tf.nn.conv2d(x_input, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
    # norm_1, mov_avg = batch_norm_layer(logits_1, tf.equal(1, 1), iteration, convolutional=True)
    log_1 = tf.nn.conv2d(x_input, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
    y_conv1 = tf.nn.relu(batch_norm_wrapper(log_1, is_training, axis=[0, 1, 2]))
    # Convolutional Layer 2:
    map_size_2 = 32
    w_conv2 = weight_variable([5, 5, map_size_1, map_size_2])
    b_conv2 = bias_variable([map_size_2])
    # h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    # h_pool2 = max_pool_2x2(h_conv2)
    log_2 = tf.nn.conv2d(y_conv1, w_conv2, strides=[1, 2, 2, 1], padding='SAME') + b_conv2
    # log_2_pool = max_pool_2x2(log_2)
    y_conv2 = tf.nn.relu(batch_norm_wrapper(log_2, is_training, axis=[0, 1, 2]))
    # y_conv2_pool = tf.nn.max_pool(y_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Convolutional Layer 3:
    map_size_3 = 64
    w_conv3 = weight_variable([4, 4, map_size_2, map_size_3])
    b_conv3 = bias_variable([map_size_3])
    # h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    # h_pool2 = max_pool_2x2(h_conv2)
    log_3 = tf.nn.conv2d(y_conv2, w_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3
    # log_3_pool = max_pool_2x2(log_3)
    y_conv3 = tf.nn.relu(batch_norm_wrapper(log_3, is_training, axis=[0, 1, 2]))
    # y_conv3_pool = tf.nn.max_pool(y_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    map_size_3_2 = 64
    w_conv3_2 = weight_variable([4, 4, map_size_3, map_size_3_2])
    b_conv3_2 = bias_variable([map_size_3_2])
    # h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    # h_pool2 = max_pool_2x2(h_conv2)
    log_3_2 = tf.nn.conv2d(y_conv3, w_conv3_2, strides=[1, 2, 2, 1], padding='SAME') + b_conv3_2
    # log_3_pool = max_pool_2x2(log_3)
    y_conv3_2 = tf.nn.relu(batch_norm_wrapper(log_3_2, is_training, axis=[0, 1, 2]))
    # y_conv3_pool = tf.nn.max_pool(y_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Fully connected layer:
    fc_input = tf.reshape(y_conv3_2, [-1, 8 * 8 * map_size_3])
    w_fc1 = weight_variable([8 * 8 * map_size_3, 1024])
    b_fc1 = bias_variable([1024])
    log_4 = tf.matmul(fc_input, w_fc1) + b_fc1
    y_fc1 = tf.nn.relu(batch_norm_wrapper(log_4, is_training, axis=[0]))
    y_fc1_drop = tf.nn.dropout(y_fc1, keep_prob)

    # fc_input = tf.reshape(y_conv3_2, [-1, 8 * 8 * map_size_3])
    w_fc2 = weight_variable([1024, 128])
    b_fc2 = bias_variable([128])
    log_5 = tf.matmul(y_fc1_drop, w_fc2) + b_fc2
    y_fc2 = tf.nn.relu(batch_norm_wrapper(log_5, is_training, axis=[0]))

    y_fc2_drop = tf.nn.dropout(y_fc2, keep_prob)

    # Dropout:
    # keep_prob = tf.placeholder(tf.float32)

    # Read out layer:
    w_fc2 = weight_variable([128, n_classes])
    b_fc2 = bias_variable([n_classes])

    y_out = tf.nn.softmax(tf.matmul(y_fc2_drop, w_fc2) + b_fc2)

    return y_out


def train_neural_network(x):  # x is the input data
    epochs = 20
    # prediction = convolutional_neural_network(x)
    prediction = new_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=prediction))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
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
                    sess.run(optimizer, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, is_training: True})
                    if k % 25 == 0:
                        print('Reached step %3d' % k, '(of 100) of train file', (file_train+1), '(of 5) with accuracy ', end='')
                        print(accuracy.eval(feed_dict={x: batch_test[0], y_: batch_test[1], keep_prob: 1.0, is_training: False}))
            # save_path = saver.save(sess, "save/saved_net.ckpt")
            # print("Saved to:", save_path)
        time_end = datetime.datetime.now()
        print("\nFinished training in", (time_end - start_time))
        print("Epochs: ", epochs)
        print("Testing...")
        file_test = 5  # the 5th element corresponds to the test file in the datasets list
        for k in range(10):
            batch_test = get_batch(k*1000, 1000, file_test)
            print("Size test:", len(batch_test[0]))
            print('Test %d accuracy =' % k, end=' ')
            print(accuracy.eval(feed_dict={x: batch_test[0], y_: batch_test[1], keep_prob: 1.0, is_training: False}))
            print("loss:", cost.eval(feed_dict={x: batch_test[0], y_: batch_test[1], keep_prob: 1.0, is_training: False}))


load_datasets()
train_neural_network(x)
