import cv2
import numpy as np


model = cv2.dnn.readNetFromTensorflow(r'C:\Users\User\Desktop\desktop\keras-tf-pb-master\models_tmp\modelVehicles9.pb')


image = cv2.imread(r'D:\Monika\VideosTest\slika_test\mil_res\310zaMonika.bmp',0)
[h,w]=np.asarray(image).shape
blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(w, h))

model.setInput(blob)

output = model.forward()

output_cls=output[0,:4,:,:]
print(output_cls[0,0,1])

