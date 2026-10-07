srcIm=r'D:\Monika\VideosTest\Frames_Utrinsko_Original\miladinovci'
dst=r'D:\Monika\VideosTest\Frames_Utrinsko\drugo'
dst1=r'D:\Monika\VideosTest\Frames_Utrinsko_Original_full\miladinovci'

import os
import shutil
list=os.listdir(srcIm)
for im_name in list:
    for im_name1 in os.listdir(dst):
        if im_name==im_name1:
            shutil.move(os.path.join(srcIm, im_name),os.path.join(dst1, im_name))
        else:
            continue

