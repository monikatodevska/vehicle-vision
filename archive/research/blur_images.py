import os
import numpy as np
import cv2

im_path_miladinovci = r'E:\Science\Monika2\sliki'
im_path_dst=r'E:\Science\Monika2\sliki_augmented'

im_path_tamara=r'E:\Science\Monika1\tamara\filtered_photos\test'
im_path_dst1=r'E:\Science\Monika1\tamara\filtered_photos\augmented'

def augment(im_path, dst_im_path):
    listOrig=os.listdir(im_path)
    kernel = np.array([
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1]
    ]) / 9
    # tamara='filtered_photos'
    # miladinovci='video2'
    for im_ind, im_name in enumerate(listOrig):

            src_image = cv2.imread(os.path.join(im_path, im_name))
            rows, col = src_image.shape[:2]
            # desno = 50
            # levo = 248

            if col==840: #tamara
                im_res = cv2.resize(src_image, (int(col / 2), int(rows / 2)), interpolation=cv2.INTER_AREA)
                rows_r, col_r = im_res.shape[:2]
                cropped_image = im_res[0:rows_r, 47:col_r - 34]
                rows_cr, col_cr = cropped_image.shape[:2]
                resized = cv2.resize(cropped_image, (512, 341), interpolation=cv2.INTER_AREA)

                # src_image = cv2.imread(os.path.join(trainOriginal, im_name))


            elif col==1344:
                cropped_image = src_image[0:rows, 0:1141]
                rows_cr, col_cr = cropped_image.shape[:2]
                resized = cv2.resize(cropped_image, (512, 341), interpolation=cv2.INTER_AREA)

            cv2.imwrite(os.path.join(dst_im_path, str(im_name[:-4]) +  '.jpg'), resized)

            resulting_image = cv2.filter2D(resized, -1, kernel)
            cv2.imwrite(os.path.join(dst_im_path, str(im_name[:-4]) + '_b' + '.jpg'), resulting_image)
            regular_flipped = cv2.flip(resized, 1)
            blured_flipped = cv2.flip(resulting_image, 1)
            # name = int(im_name[5:11]) + k
            # im_name1 = 'image' + str(name).zfill(6) + '.jpg'
            cv2.imwrite(os.path.join(dst_im_path, im_name[:-4] + '_f' + '.jpg'), regular_flipped)
            cv2.imwrite(os.path.join(dst_im_path, im_name[:-4] + '_f_b' + '.jpg'), blured_flipped)


augment(im_path_miladinovci,im_path_dst)
# augment(im_path_tamara,im_path_dst1)
