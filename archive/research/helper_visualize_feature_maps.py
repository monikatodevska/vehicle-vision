"""



import matplotlib.pyplot as plt
import numpy as np

from keras.preprocessing import image


img_path = '/Users/User/Downloads/img1700.png'

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

print(img_tensor.shape)

"""


import matplotlib.pyplot as plt
import numpy as np

path = r'C:\Users\User\Desktop\acc.txt'
data = np.loadtxt(path)
print(data)

plt.plot(data[:, 1], 'r')
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.grid()
plt.show()    # blocks execution until figure is closed
# plt.savefig(os.path.join(dst_path, 'loss.png'))  # loss.png - name of loss graph
plt.close()

