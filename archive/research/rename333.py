import os
import shutil
srcImages=r'E:\Science\Monika\anchorless_val1'
dst=r'E:\Science\Monika3\val_gt1'
dst1=r'E:\Science\Monika3\train_gt'

srcIm=r'E:\Science\Monika3\site_sliki'
dstIm=r'E:\Science\Monika3\val_sliki'
# for gt in os.listdir(srcImages):
#     # br=gt[5:11]
#     im=gt[:-4]+'.jpg'
#     print((im))
#     if os.path.exists(os.path.join(dst,gt)):
#         # shutil.move(os.path.join(dst1,gt), os.path.join(dst,gt))
#         shutil.move(os.path.join(srcIm,im), os.path.join(dstIm,im))
#
#     else:
#         continue

# srcGT=r'E:\Science\Monika3\val_gt'
# # dst=r'E:\Science\Monika1\Video2_frames_augmented_val1'
# dstannot=r'E:\Science\Monika3\val_gt1'
# # # srcImagesTam=r'E:\Science\Monika1\tamara\filtered_photos\augmented'
# empty=r'E:\Science\Monika1\prazni'
#
# for im_name in os.listdir(srcGT):
#     # br=int(im_name[5:11])
#     # a=br-1
#     # extension=im_name[-4:]
#     # # print(extension)
#     # br_n=br+1
#     # slika='image'+str(br_n).zfill(6)+extension
#
#     # anot_old=str(a)+'.txt'
#     # annot=br-1
#     # annot_new=annot+1
#     # annot_1=str(annot_new)+'.txt'
#     # print(annot_1)
#     # os.rename(os.path.join(srcImages, im_name), os.path.join(dst, slika))
#     br=int(im_name[0:-4])
#     br1=br+1
#     annot_new='image'+str(br1).zfill(6)+'.txt'
#     os.rename(os.path.join(srcGT, im_name), os.path.join(dstannot, annot_new))

# for im_name in os.listdir(sliki):
