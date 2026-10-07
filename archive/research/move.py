import os
import shutil
import cv2

GT_test=r'E:\Science\Monika4\GT_test'
im_src=r'E:\Science\Monika4\site_sliki'
im_dst=r'E:\Science\Monika4\test_sliki'
sliki_pred=r'E:\Science\Monika4\test_sliki_pred'
for im_name in os.listdir(GT_test):
    im_name1=im_name[:-4]+'.jpg'
    shutil.copy(os.path.join(im_src,im_name1), os.path.join(im_dst,im_name1))


for im_name in os.listdir(sliki_pred):
    image = cv2.imread(os.path.join(sliki_pred, im_name))
    rows, col = image.shape[:2]
    cropped_image = image[0:rows, 0:1141]
    rows_cr, col_cr = cropped_image.shape[:2]
    resized = cv2.resize(cropped_image, (512, 341), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(im_dst, im_name), resized)