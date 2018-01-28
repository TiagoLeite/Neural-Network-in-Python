from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import shutil
from PIL import Image
import numpy as np
import datetime
import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
EXPORT_DIR = './model'
if os.path.exists(EXPORT_DIR):
    shutil.rmtree(EXPORT_DIR)

iterat = 1
n_classes = 10


def read_data(start, end, pattern):
    formatted_data = [[], []]
    for j in range(36):
        for k in range(start, end):
            name = pattern % ((j + 1), (j + 1), (k + 1))
            img = Image.open(name).convert('L')
            # print(np.shape(img))
            img_array = (np.array(img).reshape(28 * 28)) / 255.0
            formatted_data[0].append(img_array)
            label = [0.0] * 36
            label[j] = 1.0
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


def batch_norm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)  # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages


def compatible_convolutional_noise_shape(Y):
    noise_shape = tf.shape(Y)
    noise_shape = noise_shape * tf.constant([1, 0, 0, 1]) + tf.constant([0, 1, 1, 0])
    return noise_shape


def no_batch_norm(y_logits, is_test, iteration, offset, convolutional=False):
    return y_logits, tf.no_op()


print("Reading mnist...")
# sess = tf.InteractiveSession()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True, validation_size=0)

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

# =========

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
    epochs = 30
    for p in range(epochs):
        for i in range(600):
            # batch_font = read_data(i*5, (i+1)*5, 'fnt/Sample%.3d/img%.3d-%.5d.png')
            batch_font = mnist.train.next_batch(100)
            learning_rate = \
                min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-(p*600+i)/decay_speed)
            if i % 100 == 0:
                train_acc = accuracy.eval(feed_dict={x: batch_font[0], y_: batch_font[1],
                                                     lr: learning_rate, keep_prob: 1.0, is_test: True})
                print('Step %3d/550 in epoch %d of %d , training accuracy %g'
                      % (i, p, epochs, train_acc))
            train_step.run(feed_dict={x: batch_font[0], y_: batch_font[1], lr: learning_rate, keep_prob: 0.75, is_test: False})
            update_ema.run(feed_dict={x: batch_font[0], y_: batch_font[1], keep_prob: 1.0, iteration: p*600+i, is_test: False})
    end = datetime.datetime.now()
    print("\nFinished training in", (end - start))
    print("\tTesting...")
    media = []
    for _ in range(20):
        batch = mnist.test.next_batch(500)
        acc = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0, is_test: True})
        print(100 * acc)
        media.append(acc)
    print('Media: ', 100 * tf.reduce_mean(media).eval())
    '''
    batch_hand = read_data(48, 55, 'handwritten/Sample%.3d/img%.3d-%.3d.png')
    res = accuracy.eval(feed_dict={x: batch_hand[0], y_: batch_hand[1], keep_prob: 1.0})
    print("Testing Hand Digit Accuracy: ", res)

    batch_font = read_data(900, 925, 'fnt/Sample%.3d/img%.3d-%.5d.png')
    res = accuracy.eval(feed_dict={x: batch_font[0], y_: batch_font[1], keep_prob: 1.0})
    print("Testing Font Digit Accuracy: ", res)
    batch_font = read_data(925, 950, 'fnt/Sample%.3d/img%.3d-%.5d.png')
    res = accuracy.eval(feed_dict={x: batch_font[0], y_: batch_font[1], keep_prob: 1.0})
    print("Testing Font Digit Accuracy: ", res)
    batch_font = read_data(950, 975, 'fnt/Sample%.3d/img%.3d-%.5d.png')
    res = accuracy.eval(feed_dict={x: batch_font[0], y_: batch_font[1], keep_prob: 1.0})
    print("Testing Font Digit Accuracy: ", res)
    batch_font = read_data(975, 1000, 'fnt/Sample%.3d/img%.3d-%.5d.png')
    res = accuracy.eval(feed_dict={x: batch_font[0], y_: batch_font[1], keep_prob: 1.0})
    print("Testing Font Digit Accuracy: ", res)'''

    W_C1 = w_conv1.eval(sess)
    B_C1 = b_conv1.eval(sess)

    W_C2 = w_conv2.eval(sess)
    B_C2 = b_conv2.eval(sess)

    W_FC1 = w_fc1.eval(sess)
    B_FC1 = b_fc1.eval(sess)

    W_FC2 = w_fc2.eval(sess)
    B_FC2 = b_fc2.eval(sess)

graph = tf.Graph()

'''
with graph.as_default():

    x_2 = tf.placeholder('float32', shape=[None, 28*28], name='input')
    x_image = tf.reshape(x_2, [-1, 28, 28, 1])

    # Convolutional Layer 1:
    W_C1 = tf.constant(W_C1, name="WCONV1")
    B_C1 = tf.constant(B_C1, name="BCONV1")
    CONV1 = tf.nn.relu(conv2d(x_image, W_C1) + B_C1)
    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    H_POOL1 = max_pool_2x2(CONV1)

    # Convolutional Layer 2:
    W_C2 = tf.constant(W_C2, name="WCONV2")
    B_C2 = tf.constant(B_C2, name="BCONV2")
    CONV2 = tf.nn.relu(conv2d(H_POOL1, W_C2) + B_C2)
    H_POOL2 = max_pool_2x2(CONV2)

    # Fully connected layer:
    W_FC1 = tf.constant(W_FC1, name="WFC1")
    B_FC1 = tf.constant(B_FC1, name="BFC1")

    FC1 = tf.reshape(H_POOL2, [-1, 7*7*64])
    FC1 = tf.nn.relu(tf.matmul(FC1, W_FC1) + B_FC1)
    # h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Read out layer:
    W_FC2 = tf.constant(W_FC2, name="WFC2")
    B_FC2 = tf.constant(B_FC2, name="BFC2")
    Y_CONV = tf.nn.softmax(tf.matmul(FC1, W_FC2) + B_FC2, name='output')
    # no need to Dropout

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        graph_def = graph.as_graph_def()
        tf.train.write_graph(graph_def, EXPORT_DIR, 'mnist_model_graph.pb', as_text=False)

        y_train = tf.placeholder('float', [None, 10])
        correct_prediction = tf.equal(tf.argmax(Y_CONV, 1), tf.argmax(y_train, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        # print("\nSave file with acc = %g" % accuracy.eval({x_2: mnist.test.images, y_train: mnist.test.labels}, sess))
        partials = []
        for _ in range(10):
            batch = mnist.test.next_batch(1000)
            res = accuracy.eval(feed_dict={x_2: batch[0], y_train: batch[1]})
            partials.append(res)
            # print("Testing Accuracy: ", res)
        print(partials)
        print("Avg:", tf.reduce_mean(partials).eval())
'''
