import os
import shutil
srcImages=r'D:\Monika\Demo_Images_Oblacno1\miladinovci'
dst=r'D:\Monika\Demo_Images_Oblacno\miladinovci'
# gt=r'D:\KlasifikacijaVozila\M-30\GT_val'

list=os.listdir(srcImages)
# lista=list[0:]
# m=33820

for im_name in list:
    br=im_name[:-4]
    ext=im_name[-4:]
    new_im=str(br).zfill(5)+ext

    print(new_im)
    #
    # gt_new='image'+str(br).zfill(6)+'.txt'
    # im_new='image'+str(br).zfill(6)+'.jpg'
    # print (im_name)
    os.rename(os.path.join(srcImages,im_name), os.path.join(dst,new_im))



