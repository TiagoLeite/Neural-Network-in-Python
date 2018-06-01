import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
from PIL import Image
import random

n_classes = 62
names_char = np.arange(0, 62)
image_size = 32, 32
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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


def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
                                input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3,
                                           name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                      name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3,
                                            name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0);
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


with tf.gfile.FastGFile('model/retrained_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name="my_graph")

with tf.Session() as sess:
    for i in tf.get_default_graph().get_operations():
        print(i.name)

    print("\tTesting...")

    output_tensor = sess.graph.get_tensor_by_name('my_graph/final_result:0')

    input_tensor = sess.graph.get_tensor_by_name('my_graph/input:0')

    print(output_tensor, input_tensor)

    image = read_tensor_from_image_file('flower.jpg',
                                        input_height=224,
                                        input_width=224,
                                        input_mean=128,
                                        input_std=128)

    # image = Image.open("flower.jpg")

    # image = image.resize((224, 224), Image.ANTIALIAS)

    # image.save('imago.jpg')

    # image = np.reshape(image, [-1, 224, 224, 3])

    # im = Image.fromarray(np.reshape(image, [224, 224, 3]))

    # im.save("your_file.jpeg")

    print(np.shape(image))

    acc = sess.run(output_tensor, feed_dict={'my_graph/input:0': image})

    print(acc)

    index = np.argmax(acc, axis=1)

    print('index:', index[0])

    '''
    class_names = ["bluebell", "buttercup", "coltsfoot", "cowslip", "crocus",
                   "daffodil", "daisy", "dandelion", "fritillary", "iris", "lilyvalley", "pansy",
                   "snowdrop", "sunflower", "tigerlily", "tulip", "windflower"] '''

    class_names = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

    print(class_names[index[0]])
