import numpy as np
from PIL import Image

image = Image.open('images/Sample028/img028-001.png').convert('L')
image = image.resize((128, 128))
image.show()
array_image = np.array(image)
print(array_image.shape)
array_image = np.reshape(array_image, [128, 128])
print(array_image.shape)
array_image = np.fliplr(array_image)
# array_image[:, :, 1] = np.fliplr(array_image[:, :, 1])
# array_image[:, :, 2] = np.fliplr(array_image[:, :, 2])
# array_image = np.reshape(array_image, [128, 128, 3])
im2 = Image.fromarray(np.uint8(array_image))
im2.show()

