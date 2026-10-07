
import os
import shutil

site_sliki=r'E:\Science\Monika3\site_sliki'
srcGT=r'E:\Science\Monika3\GroundTruthFilesAnchorless555'
dst_im=r'E:\Science\Monika3\site_sliki_GT'
# for im_name in os.listdir(site_sliki):
#     if im_name[-4:]=='.bmp':
#         im_name_n=im_name[0:11]+'.jpg'
#         os.rename(os.path.join(site_sliki,im_name), os.path.join(site_sliki,im_name_n))
for gt in os.listdir(srcGT):
    im_name=gt[:-4] +'.jpg'
    shutil.move(os.path.join(site_sliki,im_name), os.path.join(dst_im,im_name))
