import cv2
import numpy as np
import os

groundtruthfilesTrain = r'E:\Science\Monika1\site-baseline-cropped'
groundtruthfilesTrain1 = r'E:\Science\Monika1\site-baseline-cropped-resized'

im_size=(512,341)
for im_ind, im_name in enumerate(os.listdir(groundtruthfilesTrain)):
    image=cv2.imread(os.path.join(groundtruthfilesTrain, im_name))
    result=cv2.resize(image, im_size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(groundtruthfilesTrain1, im_name), result)

