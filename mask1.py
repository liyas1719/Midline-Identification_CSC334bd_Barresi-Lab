import numpy as np

from PIL import Image
import numpy as np

image = Image.open('/Users/liyas1719/Desktop/dataset/maksimages/image40.jpg').convert('L')
image_array = np.array(image)
image_array = (image_array > 200).astype(np.uint8) * 255

print(sum(image_array))
print(image_array.shape)

np.save("image40.npy", image_array)