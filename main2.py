from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import shutil
from PIL import Image
import numpy as np
import datetime
import os
import random
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
EXPORT_DIR = './model_char'
if os.path.exists(EXPORT_DIR):
    shutil.rmtree(EXPORT_DIR)

iterat = 1
n_classes = 62
names_char = np.arange(0, 62)
image_size = 32, 32


def read_data(start, end, pattern):
    formatted_data = [[], []]
    random.shuffle(names_char)
    for j in range(n_classes):
        for k in range(start, end):  # 'handwritten/$s/train_%s/train_%s_%.5d.png'
            name = pattern % (names_char[j], names_char[j], k)
            img = Image.open(name).convert('L')
            img_array = 1.0 - (np.array(img).reshape(32 * 32)) / 255.0
            formatted_data[0].append(img_array)
            label = [0.0] * n_classes
            label[names_char[j]] = 1.0
            formatted_data[1].append(label)
    return formatted_data


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.ones(shape=shape) / 10.0)  # 0.1


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def batch_norm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999,
                                                       iteration)  # adding the iteration prevents from averaging across non-existing iterations
    epsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, epsilon)
    return ybn, update_moving_averages


def batch_norm_infer(Ylogits, offset, mean, var):
    bnepsilon = 1e-5
    Ybn = tf.nn.batch_normalization(Ylogits, mean, var, offset, None, bnepsilon)
    return Ybn


def print_image(image_array, w):
    for k in range(len(image_array[0])):
        if k % w == 0:
            print('')
        # if image_array[0][k] == 1:
        #    print('0', end='')
        # else:
        #    print('1', end='')
        print(image_array[0][k], end=' ')


print("Starting...")
# sess = tf.InteractiveSession()
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 32 * 32])
y_ = tf.placeholder(tf.float32, shape=[None, n_classes])
is_test = tf.placeholder(tf.bool)
iteration = tf.placeholder(tf.int32)
lr = tf.placeholder(tf.float32)

# ========== Model ===========

x_input = tf.reshape(x, [-1, 32, 32, 1])
# Convolutional Layer 1:
map_size_1 = 32
w_conv1 = weight_variable([6, 6, 1, map_size_1])
b_conv1 = bias_variable([map_size_1])
# norm_1, mov_avg = batch_norm_layer(logits_1, tf.equal(1, 1), iteration, convolutional=True)
log_1 = tf.nn.conv2d(x_input, w_conv1, strides=[1, 1, 1, 1], padding='SAME')  # + b_conv1
y_norm, ema1 = batch_norm(log_1, is_test, iteration, b_conv1, convolutional=True)
y_conv1 = tf.nn.relu(y_norm)
# Convolutional Layer 2:
map_size_2 = 48
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

# Convolutional Layer 4:
map_size_4 = 96
w_conv4 = weight_variable([3, 3, map_size_3, map_size_4])
b_conv4 = bias_variable([map_size_4])
log_4 = tf.nn.conv2d(y_conv3, w_conv4, strides=[1, 2, 2, 1], padding='SAME')  # + b_conv3
log_4_norm, ema4 = batch_norm(log_4, is_test, iteration, b_conv4, convolutional=True)
y_conv4 = tf.nn.relu(log_4_norm)

# Convolutional Layer 5:
'''map_size_5 = 96
w_conv5 = weight_variable([3, 3, map_size_4, map_size_5])
b_conv5 = bias_variable([map_size_5])
log_5 = tf.nn.conv2d(y_conv4, w_conv5, strides=[1, 2, 2, 1], padding='SAME')  # + b_conv3
log_5_norm, ema5 = batch_norm(log_5, is_test, iteration, b_conv5, convolutional=True)
y_conv5 = tf.nn.relu(log_5_norm)'''

# Fully connected layer:
fc_input = tf.reshape(y_conv4, [-1, 4 * 4 * map_size_4])
w_fc1 = weight_variable([4 * 4 * map_size_4, 1024])
b_fc1 = bias_variable([1024])
log_6 = tf.matmul(fc_input, w_fc1) + b_fc1
log_6_norm, ema6 = batch_norm(log_6, is_test, iteration, b_fc1, convolutional=False)
y_fc1 = tf.nn.relu(log_6_norm)
# Dropout:
keep_prob = tf.placeholder(tf.float32)
y_fc1_drop = tf.nn.dropout(y_fc1, keep_prob)

# Fully connected layer 2:
w_fc2 = weight_variable([1024, 256])
b_fc2 = bias_variable([256])
log_7 = tf.matmul(y_fc1_drop, w_fc2) + b_fc2
log_7_norm, ema7 = batch_norm(log_7, is_test, iteration, b_fc2, convolutional=False)
y_fc2 = tf.nn.relu(log_7_norm)
# Dropout:
y_fc2_drop = tf.nn.dropout(y_fc2, keep_prob)

# Read out layer:
w_fc3 = weight_variable([256, n_classes])
b_fc3 = bias_variable([n_classes])

y_out = tf.nn.softmax(tf.matmul(y_fc2_drop, w_fc3) + b_fc3)

update_ema = tf.group(ema1, ema2, ema3, ema4, ema6, ema7)

# ==========================

loss_cross_entropy = 100 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_out))
train_step = tf.train.AdamOptimizer(lr).minimize(loss_cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# learning rate decay
max_learning_rate = 0.004
min_learning_rate = 0.0001
decay_speed = 2000

print("Training...")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start = datetime.datetime.now()
    epochs = 36
    epoch_size = 450
    for p in range(epochs):
        for i in range(epoch_size):
            path = 'handwritten_v2/%d/train_%d/%d.png'
            batch_font = read_data(i * 8, (i + 1) * 8, path)
            learning_rate = \
                min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(
                    - (p * epoch_size + i) / decay_speed)
            if i % 10 == 0:
                acc, loss \
                    = sess.run([accuracy, loss_cross_entropy],
                               feed_dict={x: batch_font[0], y_: batch_font[1],
                                          keep_prob: 1.0, lr: learning_rate,
                                          is_test: True})
                print('Step %3d/%d in epoch %d/%d, training accuracy %g (loss: %g)'
                      % (i, epoch_size, p, epochs, acc, loss))
            train_step.run(feed_dict={x: batch_font[0], y_: batch_font[1],
                                      keep_prob: 0.9, is_test: False, lr: learning_rate})
            update_ema.run(feed_dict={x: batch_font[0], y_: batch_font[1], keep_prob: 1.0,
                                      iteration: p * epoch_size + i,
                                      is_test: False})

    end = datetime.datetime.now()
    print("\nFinished training in", (end - start))
    print("\tTesting...")
    mean = []
    path = 'handwritten_v2/%d/train_%d/%d.png'
    for k in range(20):
        batch_hand = read_data(3600 + k * 2, 3600 + (k + 1) * 2, path)
        res = accuracy.eval(feed_dict={x: batch_hand[0], y_: batch_hand[1], keep_prob: 1.0, is_test: True})
        mean.append(res)
        print("Testing accuracy: ", res)
    print("=========\nMean accuracy   : ", tf.reduce_mean(mean).eval())

    w_c1 = w_conv1.eval(sess)
    b_c1 = b_conv1.eval(sess)

    w_c2 = w_conv2.eval(sess)
    b_c2 = b_conv2.eval(sess)

    w_c3 = w_conv3.eval(sess)
    b_c3 = b_conv3.eval(sess)

    w_c4 = w_conv4.eval(sess)
    b_c4 = b_conv4.eval(sess)

    w_fc1 = w_fc1.eval(sess)
    b_fc1 = b_fc1.eval(sess)

    w_fc2 = w_fc2.eval(sess)
    b_fc2 = b_fc2.eval(sess)

    w_fc3 = w_fc3.eval(sess)
    b_fc3 = b_fc3.eval(sess)

    variables = tf.global_variables(scope="moments")
    print(len(variables))

    for k in range(len(variables)):
        variables[k] = variables[k].eval(sess)
        # print(variables[k])

graph = tf.Graph()

with graph.as_default():
    X_2 = tf.placeholder('float32', shape=[None, 32 * 32], name='input')
    X_IMAGE = tf.reshape(X_2, [-1, 32, 32, 1])

    exps0 = tf.constant(variables[0])
    exps1 = tf.constant(variables[1])
    exps2 = tf.constant(variables[2])
    exps3 = tf.constant(variables[3])
    exps4 = tf.constant(variables[4])
    exps5 = tf.constant(variables[5])
    exps6 = tf.constant(variables[6])
    exps7 = tf.constant(variables[7])
    exps8 = tf.constant(variables[8])
    exps9 = tf.constant(variables[9])
    exps10 = tf.constant(variables[10])
    exps11 = tf.constant(variables[11])

    # Convolutional Layer 1:
    W_C1 = tf.constant(w_c1)
    B_C1 = tf.constant(b_c1)
    LOG_1 = tf.nn.conv2d(X_IMAGE, W_C1, strides=[1, 1, 1, 1], padding='SAME')  # + b_conv1
    Y_NORM = batch_norm_infer(LOG_1, B_C1, exps0, exps1)
    # Y_NORM = batch_norm_deploy(LOG_1, B_C1, convolutional=True)
    Y_C1 = tf.nn.relu(Y_NORM)
    # Convolutional Layer 2:
    W_C2 = tf.constant(w_c2)
    B_C2 = tf.constant(b_c2)
    LOG_2 = tf.nn.conv2d(Y_C1, W_C2, strides=[1, 2, 2, 1], padding='SAME')  # + b_conv2
    Y_NORM2 = batch_norm_infer(LOG_2, B_C2, exps2, exps3)
    Y_C2 = tf.nn.relu(Y_NORM2)
    # Convolutional Layer 3:
    W_C3 = tf.constant(w_c3)
    B_C3 = tf.constant(b_c3)
    LOG_3 = tf.nn.conv2d(Y_C2, W_C3, strides=[1, 2, 2, 1], padding='SAME')  # + b_conv3
    Y_NORM3 = batch_norm_infer(LOG_3, B_C3, exps4, exps5)
    Y_C3 = tf.nn.relu(Y_NORM3)
    # Convolutional Layer 4:
    W_C4 = tf.constant(w_c4)
    B_C4 = tf.constant(b_c4)
    LOG_4 = tf.nn.conv2d(Y_C3, W_C4, strides=[1, 2, 2, 1], padding='SAME')  # + b_conv3
    Y_NORM4 = batch_norm_infer(LOG_4, B_C4, exps6, exps7)
    Y_C4 = tf.nn.relu(Y_NORM4)
    # Fully connected layer:
    W_FC1 = tf.constant(w_fc1)
    B_FC1 = tf.constant(b_fc1)
    FC_INPUT = tf.reshape(Y_C4, [-1, 4 * 4 * map_size_4])
    LOG_4 = tf.matmul(FC_INPUT, W_FC1)
    Y_NORM4 = batch_norm_infer(LOG_4, B_FC1, exps8, exps9)
    Y_FC1 = tf.nn.relu(Y_NORM4)
    # Fully connected layer 2:
    W_FC2 = tf.constant(w_fc2)
    B_FC2 = tf.constant(b_fc2)
    LOG_5 = tf.matmul(Y_FC1, W_FC2)
    Y_NORM5 = batch_norm_infer(LOG_5, B_FC2, exps10, exps11)
    Y_FC2 = tf.nn.relu(Y_NORM5)
    # Read out layer:
    W_FC3 = tf.constant(w_fc3)
    B_FC3 = tf.constant(b_fc3)
    Y_OUT = tf.nn.softmax(tf.matmul(Y_FC2, W_FC3) + B_FC3, name='output')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        graph_def = graph.as_graph_def()
        tf.train.write_graph(graph_def, EXPORT_DIR, 'mnist_model_graph.pb', as_text=False)
        print("Saved model successfully")

        '''
        y_ = tf.placeholder('float', [None, n_classes])
        correct_prediction = tf.equal(tf.argmax(Y_OUT, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        print("Final test:")

        mean = []
        path = 'handwritten/%s/train_%s/train_%s_%.5d.png'
        for k in range(20):
            batch_hand = read_data(2000 + k * 20, 2000 + (k + 1) * 20, path)
            res = accuracy.eval(feed_dict={X_2: batch_hand[0], y_: batch_hand[1]})
            mean.append(res)
            print("Testing accuracy: ", res)
        print("Mean accuracy   : ", tf.reduce_mean(mean).eval())'''
