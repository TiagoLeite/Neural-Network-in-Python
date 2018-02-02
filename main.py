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
is_test = tf.placeholder(tf.bool)
lr = tf.placeholder(tf.float32)
iteration = tf.placeholder(tf.int32)
keep_prob = tf.placeholder(tf.float32)
datasets = []



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


def batch_norm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)  # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    global model_mean, model_var
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])

    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages


def batch_norm_infer(Ylogits, offset, mean, var):
    bnepsilon = 1e-5
    Ybn = tf.nn.batch_normalization(Ylogits, mean, var, offset, None, bnepsilon)
    return Ybn


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


# ===== Model =====

x_input = tf.reshape(x, [-1, 28, 28, 1])
# Convolutional Layer 1:
map_size_1 = 32
w_conv1 = weight_variable([6, 6, 1, map_size_1])
b_conv1 = bias_variable([map_size_1])
# norm_1, mov_avg = batch_norm_layer(logits_1, tf.equal(1, 1), iteration, convolutional=True)
log_1 = tf.nn.conv2d(x_input, w_conv1, strides=[1, 1, 1, 1], padding='SAME')  # + b_conv1
y_norm, ema1 = batch_norm(log_1, is_test, iteration, b_conv1, convolutional=True)
y_conv1 = tf.nn.relu(y_norm)
# Convolutional Layer 2:
map_size_2 = 32
w_conv2 = weight_variable([5, 5, map_size_1, map_size_2])
b_conv2 = bias_variable([map_size_2])
log_2 = tf.nn.conv2d(y_conv1, w_conv2, strides=[1, 2, 2, 1], padding='SAME')  # + b_conv2
log_2_norm, ema2 = batch_norm(log_2, is_test, iteration, b_conv2, convolutional=True)
y_conv2 = tf.nn.relu(log_2_norm)

# Convolutional Layer 3:
map_size_3 = 64
w_conv3 = weight_variable([4, 4, map_size_2, map_size_3])
b_conv3 = bias_variable([map_size_3])
log_3 = tf.nn.conv2d(y_conv2, w_conv3, strides=[1, 2, 2, 1], padding='SAME')  # + b_conv3
log_3_norm, ema3 = batch_norm(log_3, is_test, iteration, b_conv3, convolutional=True)
y_conv3 = tf.nn.relu(log_3_norm)

# Fully connected layer:
fc_input = tf.reshape(y_conv3, [-1, 7 * 7 * map_size_3])
w_fc1 = weight_variable([7 * 7 * map_size_3, 256])
b_fc1 = bias_variable([256])
log_4 = tf.matmul(fc_input, w_fc1) + b_fc1

log_4_norm, ema4 = batch_norm(log_4, is_test, iteration, b_fc1, convolutional=False)
y_fc1 = tf.nn.relu(log_4_norm)

# Dropout:
keep_prob = tf.placeholder(tf.float32)
y_fc1_drop = tf.nn.dropout(y_fc1, keep_prob)

# Read out layer:
w_fc2 = weight_variable([256, n_classes])
b_fc2 = bias_variable([n_classes])

y_out = tf.nn.softmax(tf.matmul(y_fc1_drop, w_fc2) + b_fc2)

update_ema = tf.group(ema1, ema2, ema3, ema4)


# =========

load_datasets()

epochs = 20
loss_cross_entropy = 100*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_out))
train_step = tf.train.AdamOptimizer(lr).minimize(loss_cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# learning rate decay
max_learning_rate = 0.02
min_learning_rate = 0.0001
decay_speed = 160

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
                batch_font = get_batch(100 * k, 100, file_train)  # gets the next 50 train images
                train_step.run(
                    feed_dict={x: batch_font[0], y_: batch_font[1], lr: learning_rate, keep_prob: 0.75, is_test: False})
                update_ema.run(feed_dict={x: batch_font[0], y_: batch_font[1], keep_prob: 1.0, iteration: p * 600 + i,
                                          is_test: False})
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

