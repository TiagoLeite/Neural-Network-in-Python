import os
import sys
from PIL import Image

size = 28, 28


def resize_image(input_image_path, output_image_path):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    print('The original image size is {wide} wide x {height} '
          'high'.format(wide=width, height=height))

    resized_image = original_image.resize(size)
    width, height = resized_image.size
    print('The resized image size is {wide} wide x {height} '
          'high\n'.format(wide=width, height=height))
    # resized_image.show()
    resized_image.save(output_image_path)

for k in range(36):
    for p in range(55):
        name = 'handwritten/Sample%.3d/img%.3d-%.3d.png' % ((k+1), (k+1), (p+1))
        # print(name)
        resize_image(name, name)

'''for k in range(1164):
    name = 'bananas/{name}.jpg'.format(name=(k+1))
    resize_image(name, name)
'''
