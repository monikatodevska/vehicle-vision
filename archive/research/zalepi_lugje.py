import os
import numpy as np
import cv2
import random
from PIL import Image
import random

path_pedestrians=r'\\192.168.1.133\Monika\Pedestrians_Data\CityPersons\Images\seq03'
annot_path=r'\\192.168.1.133\Monika\Pedestrians_Data\CityPersons\Annotations\seq03'
mil=r'\\192.168.1.133\Monika\VideosTest\Frames_morning_5fps\miladinovci'
dst_images=r'\\192.168.1.133\Monika\Pedestrians_Data\Miladinovci_Lepeni21_2\Images\mil'
dst_annot=r'\\192.168.1.133\Monika\Pedestrians_Data\Miladinovci_Lepeni21_2\Annotations\mil'
im_width_new=469
im_height_new=281
if not os.path.exists(dst_images):
    os.makedirs(dst_images,exist_ok=True)
if not os.path.exists(dst_annot):
    os.makedirs(dst_annot,exist_ok=True)
images_mil = []  # array of normalized multidiemnsional distance maps (frequencies: 20 - 120MHz)
im_mil_names = [fname for fname in os.listdir(mil) if fname[-4:] == '.bmp']

# for im_mil_name in enumerate(im_mil_names):
#         # load image
#     im = cv2.imread(os.path.join(mil, im_mil_name), 0)
#         # if im.shape[0] != 512 or im.shape[1] != 512:
#         #     print(os.path.join(path_images, im_name))
#         #     continue
#         # im = np.uint16(im)
#     images_mil.append(im)

images = []  # array of normalized multidiemnsional distance maps (frequencies: 20 - 120MHz)
im_names = os.listdir(path_pedestrians)
mil_im_names=[x for x in os.listdir(mil) if x[-4:]=='.bmp']
cnt=1
i=700
for ind,im_name in enumerate(im_names):
    # load image
    print(im_name)
    # image_lepenje = str(cnt).zfill(6) + '.bmp'
    # print(image_lepenje)
    if not os.path.exists(os.path.join(annot_path,im_name[:-4]+'.txt')):
        continue
    i+=1
    image_lepenje_im=cv2.imread(os.path.join(mil,mil_im_names[i]),0)

    image_cr_l = image_lepenje_im[45:image_lepenje_im.shape[0] - 45, 0:image_lepenje_im.shape[1]]
    # rows_cr, col_cr = image_cr_l.shape[:2]
    resized_l = cv2.resize(image_cr_l, (im_width_new, im_height_new), interpolation=cv2.INTER_AREA)

    cnt+=1
    image = cv2.imread(os.path.join(path_pedestrians,im_name), 0)
    # cv2.imshow('slika',image)
    # if im.shape[0] != 512 or im.shape[1] != 512:
    #     print(os.path.join(path_images, im_name))
    #     continue
    # im = np.uint16(im)
    # images.append(im)
    flag_ima_objekt=False
    image_cr = image[48:image.shape[0] - 48, 0:image.shape[1]]
    rows_cr, col_cr = image_cr.shape[:2]
    resized = cv2.resize(image_cr, (im_width_new, im_height_new), interpolation=cv2.INTER_AREA)
# resized=np.reshape(resized,(625,375,1))
# images.append(resized)
    HScale = im_height_new / rows_cr
    WScale = im_width_new / col_cr
    import math
# resized = resized.reshape(resized.shape[0], resized.shape[1], 1)
    bboxes = np.loadtxt(os.path.join(annot_path, im_name[:-4]+'.txt'), delimiter=',', ndmin=2).astype(np.int)
    for bbox in bboxes:
        object = [min(int((bbox[0] - 48) * HScale), resized.shape[0]), min(int(bbox[1] * WScale), resized.shape[1]), min(int((bbox[2] - 48) * HScale), resized.shape[0]),
                        min(int(bbox[3] * WScale), resized.shape[1]), 1]
        if (object[2]-object[0]) > 35 and (object[2]-object[0]) <51:
            x=random.randint(1+35, 468-35)
            y=random.randint(109, 190)
            half_h=math.ceil((object[2]-object[0])/2)
            half_w=math.ceil((object[3]-object[1])/2)
            # print(half_h)

            # print(y-half_h)
            # print(y+half_h)
            # print(x-half_w)
            # print(x+half_w)
            # print(object[2]-object[0])
            # print(object[3]-object[1])

            resized_l[y-half_h:y-half_h+(object[2]-object[0]),x-half_w:x-half_w+(object[3]-object[1])]=resized[object[0]:object[2],object[1]:object[3]]
            flag_ima_objekt=True
            objekt = [y - half_h, x - half_w, y - half_h + (object[2] - object[0]), x - half_w + (object[3] - object[1]), 1]
            break
        elif (object[2]-object[0]) > 55 and (object[2]-object[0]) <70:
            x = random.randint(1+35, 468-35)
            y = random.randint(190, 281-35)
            half_h = math.ceil((object[2] - object[0]) / 2)
            half_w = math.ceil((object[3] - object[1]) / 2)
            resized_l[y - half_h:y - half_h+(object[2]-object[0]), x - half_w:x - half_w+(object[3]-object[1])] = resized[object[0]:object[2], object[1]:object[3]]
            flag_ima_objekt=True
            objekt=[y-half_h,x-half_w,y - half_h+(object[2]-object[0]),x - half_w+(object[3]-object[1]),1]
            break
    if flag_ima_objekt:
        cv2.imwrite(os.path.join(dst_images,im_name), resized_l)
        np.savetxt(os.path.join(dst_annot,im_name[:-4]+'.txt'),objekt,fmt='%i',delimiter=',')

    # cv2.imshow('slika', resized_l)
    # cv2.waitKey(0)