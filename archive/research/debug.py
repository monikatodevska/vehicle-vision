import numpy as np

import os
import cv2

GT_root=r'D:\Monika\Cyclists_Animals_Training_New1_Renamed\GT'
IM_root=r'D:\Monika\Cyclists_Animals_Training_New1_Renamed\Images'


gt_filenames=os.listdir(os.path.join(GT_root))

for gt_filename in gt_filenames:

    image=cv2.imread(os.path.join(IM_root,gt_filename[:-4]+'.bmp'),0)
    if image is None:
        print(os.path.join(IM_root,gt_filename[:-4]+'.bmp'))
        print(GT_root,gt_filename)
        continue
