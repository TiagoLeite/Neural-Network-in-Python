import os
from PIL import Image
limit = 2473

size = 28, 28


def resize_image(input_image_path, output_image_path):
    original_image = Image.open(input_image_path)
    # width, height = original_image.size
    # print('The original image size is {wide} wide x {height} '
    #      'high'.format(wide=width, height=height))
    resized_image = original_image.resize(size, resample=Image.LANCZOS)
    # width, height = resized_image.size
    # print('The resized image size is {wide} wide x {height} '
    #       'high\n'.format(wide=width, height=height))
    # resized_image.show()
    resized_image.save(output_image_path)


def crop_image(img_path, xy, scale_factor):
    """Crop the image around the tuple xy
    Inputs:
    -------
    img: Image opened with PIL.Image
    xy: tuple with relative (x,y) position of the center of the cropped image
        x and y shall be between 0 and 1
    scale_factor: the ratio between the original image's size and the cropped image's size
    """
    img = Image.open(img_path)
    center = (img.size[0] * xy[0], img.size[1] * xy[1])
    new_size = (img.size[0] / scale_factor, img.size[1] / scale_factor)
    left = max (0, (int) (center[0] - new_size[0] / 2))
    right = min (img.size[0], (int) (center[0] + new_size[0] / 2))
    upper = max (0, (int) (center[1] - new_size[1] / 2))
    lower = min (img.size[1], (int) (center[1] + new_size[1] / 2))
    cropped_img = img.crop((left, upper, right, lower))
    cropped_img.save(img_path)


for k in range(65, 91):
    hex_val = format(k, 'x')
    print('Resizing %d' % k)
    for j in range(limit):
        name = 'handwritten/'+hex_val+'/train_'+hex_val+'/train_'+hex_val+'_%.5d.png' % j
        # crop_image(name, [0.5, 0.5], 1.5)
        resize_image(name, name)

for k in range(48, 58):
    hex_val = format(k, 'x')
    print('Cropping %d' % k)
    for j in range(limit):
        name = 'handwritten/'+hex_val+'/train_'+hex_val+'/train_'+hex_val+'_%.5d.png' % j
        crop_image(name, [0.5, 0.5], 1.5)
        # resize_image(name, name)

for k in range(48, 58):
    hex_val = format(k, 'x')
    print('Cropping %d' % k)
    for j in range(limit):
        name = 'handwritten/'+hex_val+'/train_'+hex_val+'/train_'+hex_val+'_%.5d.png' % j
        # crop_image(name, [0.5, 0.5], 1.5)
        resize_image(name, name)

