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
n_classes = 36
names = np.arange(1, n_classes+1)


def read_data(start, end, pattern):
    formatted_data = [[], []]
    random.shuffle(names)
    for j in range(n_classes):
        for k in range(start, end):
            name = pattern % ((names[j]), (names[j]), (k + 1))
            img = Image.open(name).convert('L')
            # print(np.shape(img))
            img_array = (np.array(img).reshape(28 * 28)) / 255.0
            formatted_data[0].append(img_array)
            label = [0.0] * n_classes
            label[names[j]-1] = 1.0
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
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)  # adding the iteration prevents from averaging across non-existing iterations
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

print("Starting...")
# sess = tf.InteractiveSession()
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, n_classes])
is_test = tf.placeholder(tf.bool)
iteration = tf.placeholder(tf.int32)
lr = tf.placeholder(tf.float32)

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

# ===============

loss_cross_entropy = 100*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_out))
train_step = tf.train.AdamOptimizer(lr).minimize(loss_cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# learning rate decay
max_learning_rate = 0.02
min_learning_rate = 0.0001
decay_speed = 1600

print("Training...")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start = datetime.datetime.now()
    epochs = 20
    cont = 0
    for p in range(epochs):
        learning_rate = \
            min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(- cont / decay_speed)
        cont += 1
        '''for i in range(300):
            batch_font = read_data(i*3, (i+1)*3, 'fnt/Sample%.3d/img%.3d-%.5d.png')
            # batch_font = mnist.train.next_batch(100)
            if i % 99 == 0:
                train_acc = accuracy.eval(feed_dict={x: batch_font[0], y_: batch_font[1],
                                                     keep_prob: 1.0, lr: learning_rate, is_test: True})
                print('Step %3d/300 of %d/%d, font digit training accuracy %g'
                      % (i, p, epochs, train_acc))
            train_step.run(feed_dict={x: batch_font[0], y_: batch_font[1],
                                      keep_prob: 0.75, is_test: False, lr: learning_rate})
            update_ema.run(feed_dict={x: batch_font[0], y_: batch_font[1], keep_prob: 1.0, iteration: cont,
                                      is_test: False})'''
        for k in range(10):
            for j in range(16):
                # batch = mnist.train.next_batch(100)
                batch_hand = read_data(j*3, (j+1)*3, 'handwritten/Sample%.3d/img%.3d-%.3d.png')
                if j == 0:
                    train_acc = accuracy.eval(feed_dict={x: batch_hand[0], y_: batch_hand[1],
                                                         keep_prob: 1.0, lr: learning_rate, is_test: True})
                    print('Step %3d, hand digit training accuracy %g' % (p, train_acc))
                train_step.run(feed_dict={x: batch_hand[0], y_: batch_hand[1],
                                          keep_prob: 0.75, is_test: False, lr: learning_rate})
                update_ema.run(feed_dict={x: batch_hand[0], y_: batch_hand[1], keep_prob: 1.0, iteration: cont,
                                          is_test: False})

    end = datetime.datetime.now()
    print("\nFinished training in", (end - start))
    print("\tTesting...")

    batch_hand = read_data(48, 55, 'handwritten/Sample%.3d/img%.3d-%.3d.png')
    res = accuracy.eval(feed_dict={x: batch_hand[0], y_: batch_hand[1], keep_prob: 1.0, is_test: True})
    print("Testing Hand Digit Accuracy: ", res)

    '''batch_font = read_data(900, 925, 'fnt/Sample%.3d/img%.3d-%.5d.png')
    res = accuracy.eval(feed_dict={x: batch_font[0], y_: batch_font[1], keep_prob: 1.0, is_test: True})
    print("Testing Font Digit Accuracy: ", res)
    batch_font = read_data(925, 950, 'fnt/Sample%.3d/img%.3d-%.5d.png')
    res = accuracy.eval(feed_dict={x: batch_font[0], y_: batch_font[1], keep_prob: 1.0, is_test: True})
    print("Testing Font Digit Accuracy: ", res)
    batch_font = read_data(950, 975, 'fnt/Sample%.3d/img%.3d-%.5d.png')
    res = accuracy.eval(feed_dict={x: batch_font[0], y_: batch_font[1], keep_prob: 1.0, is_test: True})
    print("Testing Font Digit Accuracy: ", res)
    batch_font = read_data(975, 1000, 'fnt/Sample%.3d/img%.3d-%.5d.png')
    res = accuracy.eval(feed_dict={x: batch_font[0], y_: batch_font[1], keep_prob: 1.0, is_test: True})
    print("Testing Font Digit Accuracy: ", res)'''

    w_c1 = w_conv1.eval(sess)
    b_c1 = b_conv1.eval(sess)

    w_c2 = w_conv2.eval(sess)
    b_c2 = b_conv2.eval(sess)

    w_c3 = w_conv3.eval(sess)
    b_c3 = b_conv3.eval(sess)

    w_fc1 = w_fc1.eval(sess)
    b_fc1 = b_fc1.eval(sess)

    w_fc2 = w_fc2.eval(sess)
    b_fc2 = b_fc2.eval(sess)

    variables = tf.global_variables(scope="moments")
    print(len(variables))

    for k in range(len(variables)):
        variables[k] = variables[k].eval(sess)
        # print(variables[k])


graph = tf.Graph()

with graph.as_default():

    X_2 = tf.placeholder('float32', shape=[None, 28 * 28], name='input')
    X_IMAGE = tf.reshape(X_2, [-1, 28, 28, 1])

    exps0 = tf.constant(variables[0])
    exps1 = tf.constant(variables[1])
    exps2 = tf.constant(variables[2])
    exps3 = tf.constant(variables[3])
    exps4 = tf.constant(variables[4])
    exps5 = tf.constant(variables[5])
    exps6 = tf.constant(variables[6])
    exps7 = tf.constant(variables[7])

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
    Y_NORM2 = batch_norm_infer(LOG_2, B_C2, exps2,  exps3)
    # Y_NORM2 = batch_norm_deploy(LOG_2, B_C2, convolutional=True)
    Y_C2 = tf.nn.relu(Y_NORM2)
    # Convolutional Layer 3:
    W_C3 = tf.constant(w_c3)
    B_C3 = tf.constant(b_c3)
    LOG_3 = tf.nn.conv2d(Y_C2, W_C3, strides=[1, 2, 2, 1], padding='SAME')  # + b_conv3
    Y_NORM3 = batch_norm_infer(LOG_3, B_C3, exps4,  exps5)
    # Y_NORM3 = batch_norm_deploy(LOG_3, B_C3, convolutional=True)
    Y_C3 = tf.nn.relu(Y_NORM3)
    # Fully connected layer:
    W_FC1 = tf.constant(w_fc1)
    B_FC1 = tf.constant(b_fc1)
    FC_INPUT = tf.reshape(Y_C3, [-1, 7 * 7 * map_size_3])
    LOG_4 = tf.matmul(FC_INPUT, W_FC1)
    Y_NORM4 = batch_norm_infer(LOG_4, B_FC1, exps6,  exps7)
    Y_FC1 = tf.nn.relu(Y_NORM4)
    # Read out layer:
    W_FC2 = tf.constant(w_fc2)
    B_FC2 = tf.constant(b_fc2)
    Y_OUT = tf.nn.softmax(tf.matmul(Y_FC1, W_FC2) + B_FC2, name='output')

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        graph_def = graph.as_graph_def()
        tf.train.write_graph(graph_def, EXPORT_DIR, 'mnist_model_graph.pb', as_text=False)

        print("Saved model successfully")

        y_ = tf.placeholder('float', [None, 36])
        correct_prediction = tf.equal(tf.argmax(Y_OUT, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        print("Final test:")
        batch_hand = read_data(48, 55, 'handwritten/Sample%.3d/img%.3d-%.3d.png')
        res = accuracy.eval(feed_dict={X_2: batch_hand[0], y_: batch_hand[1]})
        print("Testing Hand Digit Accuracy: ", res)
        batch_font = read_data(900, 925, 'fnt/Sample%.3d/img%.3d-%.5d.png')
        res = accuracy.eval(feed_dict={X_2: batch_font[0], y_: batch_font[1]})
        print("Testing Font Digit Accuracy: ", res)
        batch_font = read_data(925, 950, 'fnt/Sample%.3d/img%.3d-%.5d.png')
        res = accuracy.eval(feed_dict={X_2: batch_font[0], y_: batch_font[1]})
        print("Testing Font Digit Accuracy: ", res)
        batch_font = read_data(950, 975, 'fnt/Sample%.3d/img%.3d-%.5d.png')
        res = accuracy.eval(feed_dict={X_2: batch_font[0], y_: batch_font[1]})
        print("Testing Font Digit Accuracy: ", res)
        batch_font = read_data(975, 1000, 'fnt/Sample%.3d/img%.3d-%.5d.png')
        res = accuracy.eval(feed_dict={X_2: batch_font[0], y_: batch_font[1]})
        print("Testing Font Digit Accuracy: ", res)
