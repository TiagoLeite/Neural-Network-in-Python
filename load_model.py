import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Reading test mnist...")
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.gfile.FastGFile('model/mnist_model_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name="my_graph")

with tf.Session() as sess:
    # for i in tf.get_default_graph().get_operations():
        # print(i.name)
    image = Image.open('digit.png').convert('L')
    # image = image.resize((128, 128))
    # image.show()
    array_image = np.array(image)
    array_image = np.reshape(array_image, [1, 28*28])
    print(array_image)
    for k in range(array_image[0].size):
        if array_image[0][k] > 128:
            array_image[0][k] = 0
        else:
            array_image[0][k] = 1
    print(array_image)
    softmax_tensor = sess.graph.get_tensor_by_name('my_graph/output:0')
    input_tensor = sess.graph.get_tensor_by_name('my_graph/input:0')
    # print(softmax_tensor, input_tensor)
    y_ = tf.placeholder('float32', [None, 10])
    correct_prediction = tf.equal(tf.argmax(softmax_tensor, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    batch = mnist.test.next_batch(100)
    prediction = sess.run(softmax_tensor, feed_dict={'my_graph/input:0': array_image})
    # acc = sess.run(accuracy, feed_dict={'my_graph/input:0': batch[0], y_: batch[1]})
    print(prediction)

    ''' y_train = tf.placeholder('float32', [None, 10])
    correct_prediction = tf.equal(tf.argmax(softmax_tensor, 1), tf.argmax(y_train, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float32'))
    res = accuracy.eval(feed_dict={batch[0], batch[1]})
    print(res)'''
