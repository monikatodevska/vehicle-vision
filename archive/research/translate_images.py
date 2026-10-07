#2171
import cv2
import helper_anchorless_mc
import numpy as np
import os
src=r'D:\Monika\VideosTest\Frames_morning_5fps\miladinovci'
dst=r'D:\Monika\VideosTest\proba-slika\miladinovci'
im_names=os.listdir(src)
im_indexes=[59,95,160,161,350,409,413,760,873,903,967,1040]
for ind, im_name in enumerate(im_names):
    x=2
    if ind in im_indexes:
        image=cv2.imread(os.path.join(src,im_name))
        cv2.imwrite(os.path.join(dst,im_name),image)
        while (x<=6):
            image_t=helper_anchorless_mc.shiftImageHorizontally(image,x)
            cv2.imwrite(os.path.join(dst,im_name[:-4]+'_'+str(x)+'.bmp'),image_t)
            x+=2

