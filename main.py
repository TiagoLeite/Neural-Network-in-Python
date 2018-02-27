import tensorflow as tf
import os
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
IMG_HEIGHT = 32
IMG_WIDTH = 32
MIN_SIZE = 1
MAX_SIZE = 5
NUM_IMAGES = 10000
NUM_OBJECTS = 1

bboxes = np.zeros((NUM_IMAGES, NUM_OBJECTS, 4))
imgs = np.zeros((NUM_IMAGES, IMG_WIDTH, IMG_HEIGHT))

for i_image in range(NUM_IMAGES):
    for i_object in range(NUM_OBJECTS):
        x = np.random.randint(0, IMG_WIDTH-MAX_SIZE+1)
        y = np.random.randint(0, IMG_HEIGHT-MAX_SIZE+1)
        width = x + np.random.randint(MIN_SIZE, MAX_SIZE)
        height = y + np.random.randint(MIN_SIZE, MAX_SIZE)
        imgs[i_image, x:x+width, y:y+height] = 1
        bboxes[i_image, i_object] = [x, y, width, height]

i = 0
plt.imshow(imgs[i].T, cmap="Accent", interpolation='none', origin='lower', extent=[0, IMG_WIDTH, 0, IMG_HEIGHT])

x_image = (imgs.reshape(NUM_IMAGES, -1) - np.mean(imgs)) / np.std(imgs)
y_bboxes = bboxes.reshape(NUM_IMAGES, -1) / np.max((IMG_HEIGHT, IMG_WIDTH))

print(x_image.shape)

train_size = int(0.8 * NUM_IMAGES)
x_train = x_image[:train_size]
x_test = x_image[train_size:]
y_train = y_bboxes[:train_size]
y_test = y_bboxes[train_size:]

x = tf.placeholder(tf.float32, shape=[None, IMG_WIDTH * IMG_HEIGHT])
y_ = tf.placeholder(tf.float32, shape=[None, NUM_OBJECTS*4])

# Model:
print(x.shape)
x_input = tf.reshape(x, [-1, IMG_WIDTH, IMG_HEIGHT, 1])

w_conv1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 16], stddev=0.1))
b_conv1 = tf.Variable(tf.ones(shape=[16])/10.0)
y_conv1 = tf.nn.relu(tf.nn.conv2d(x_input, w_conv1, strides=[1, 2, 2, 1], padding="SAME") + b_conv1)

w_conv2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 16, 32], stddev=0.1))
b_conv2 = tf.Variable(tf.ones(shape=[32])/10.0)
y_conv2 = tf.nn.relu(tf.nn.conv2d(y_conv1, w_conv2, strides=[1, 2, 2, 1], padding="SAME"))

fc_input = tf.reshape(y_conv2, [-1, 8 * 8 * 32])
w_fc = tf.Variable(tf.truncated_normal(shape=[8 * 8 * 32, 128], stddev=0.1))
b_fc = tf.Variable(tf.ones(shape=[128])/10.0)

y_fc = tf.nn.relu(tf.matmul(fc_input, w_fc) + b_fc)

w_out = tf.Variable(tf.truncated_normal(shape=[128, 4], stddev=0.1))
b_out = tf.Variable(tf.ones(shape=[4])/10.0)

y_out = tf.matmul(y_fc, w_out)+b_out


# loss_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_out))

loss_cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(y_, y_out), axis=[1]))
train_step = tf.train.AdamOptimizer().minimize(loss_cross_entropy)
accuracy = tf.reduce_sum(tf.squared_difference(y_, y_out))

epochs = 5

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        print("Started epoch ", epoch)
        for k in range(int(0.8 * NUM_IMAGES/100)):
            x_data = x_train[k:100*(k+1)]
            y_data = y_train[k:100*(k+1)]
            # print(x_data.shape)
            # print(y_data.shape)
            _, loss = sess.run([train_step, loss_cross_entropy], feed_dict={x: x_data, y_: y_data})
            if k % 10 == 0:
                print(loss)
                # print(accuracy.eval(feed_dict={x: x_data, y_: y_data}))

    loss = sess.run(loss_cross_entropy, feed_dict={x: x_test, y_: y_test})
    print(x_test.shape)
    print("Test loss:", loss)

    i = 0

    b = sess.run(y_out, feed_dict={x: x_train, y_: y_train})

    print(b[0])

    for k in range(10):
        plt.imshow(imgs[k].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, IMG_WIDTH, 0, IMG_HEIGHT])
        plt.gca().add_patch(matplotlib.patches.Rectangle((b[k][0]*IMG_HEIGHT,
                                                         b[k][1]*IMG_HEIGHT),
                                                         b[k][2]*IMG_HEIGHT,
                                                         b[k][3]*IMG_HEIGHT, ec='r', fc='none'))
        plt.show()

'''for bbox in bboxes[i]:
    print(bbox[0], bbox[1], bbox[2], bbox[3])
    plt.gca().add_patch(mpatches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='r', fc='none'))
plt.show()'''













