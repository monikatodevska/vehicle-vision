import os
import numpy as np
import cv2
import shutil
src_images=r'D:\Monika\Results\vinf_400_cls_reg_finetune_sonegativni1_sokoordinati_prr\kamioni\drugo'
# src_images=r'D:\Monika\Results\vinf_400_cls_reg_finetune_sonegativni1_sokoordinati_prr\kamioni\drugo'
dst_images=r'D:\Monika\Results\vinf_400_cls_reg_finetune_sonegativni1_sokoordinati_kamioni'
# dst_images=r'D:\Monika\Results\vinf_400_cls_reg_finetune_sonegativni1_sokoordinati_kamioni_postprocessing'
dst_zapisi=r'D:\Monika\Results\proba'
src_filenames=[filename for filename in os.listdir(src_images)]
dst_filenames=[filename for filename in os.listdir(dst_images) if filename[-4:]=='.bmp']
# dst_filenames_txt=[filename for filename in os.listdir(dst_images) if filename[-4:]=='.bmp']
print(dst_filenames)

for i,im_name in enumerate(dst_filenames):
    txt_name=im_name[:-4]+ '.txt'
    print(os.path.join(dst_images,im_name))
    print(os.path.join(dst_zapisi,src_filenames[i]))
    shutil.copy(os.path.join(dst_images,im_name), os.path.join(dst_zapisi,src_filenames[i]))
    # print (src_filenames[i][:-4])
    shutil.copy(os.path.join(dst_images,txt_name), os.path.join(dst_zapisi,src_filenames[i][:-4]+'.txt'))

