import numpy as np

from PIL import Image
import numpy as np

image = Image.open('/workspaces/Midline-Identification_CSC334bd_Barresi-Lab/images/image43.jpg').convert('L')
image_array = np.array(image)
image_array = (image_array > 200).astype(np.uint8) * 255

print(sum(image_array))
print(image_array.shape)

np.save("image43.npy", image_array)