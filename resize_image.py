import os
import sys
from PIL import Image

size = 28, 28


def resize_image(input_image_path, output_image_path):
    original_image = Image.open(input_image_path)
    # width, height = original_image.size
    # print('The original image size is {wide} wide x {height} '
    #      'high'.format(wide=width, height=height))

    resized_image = original_image.resize(size, resample=Image.LANCZOS)
    width, height = resized_image.size
    # print('The resized image size is {wide} wide x {height} '
    #       'high\n'.format(wide=width, height=height))
    # resized_image.show()
    resized_image.save(output_image_path)

name = 'handwritten/Sample%.3d/img%.3d-%.3d.png' % ((1), (1), (1))

for k in range(65, 91):
    a = format(k, 'x')
    print(a, end=' ')


# resize_image(name, name)

