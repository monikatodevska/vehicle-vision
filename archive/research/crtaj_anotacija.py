import os
import cv2
import numpy as np
slika=r'E:\Science\Monika1\Novi\proba-sliki'
anot=r'E:\Science\Monika1\Novi\proba-anot'

im_names=os.listdir(slika)

#
# for im_name in im_names:
#     print(im_name)
#     print([im_name[:-4]])
#     objects = []
#     image = cv2.imread(os.path.join(slika, im_name))
#     for annot_name in (os.listdir(anot)):
#         if (im_name[:-4] + '_') in annot_name or (im_name[:-4] + '.txt') in annot_name:
#             object = np.loadtxt(os.path.join(anot, annot_name), delimiter=',')
#
#             object = [int(el) for el in object]
#             print(object)
#
#             # object[1] = object[1]
#             # object[3] = object[3]
#             # object = [int(np.round(object[0] )), int(np.round(object[1])), int(np.round(object[2])), int(np.round(object[3]))]
#             objects.append(object)
#             print(object)
#     for obj in objects:
#         cv2.rectangle(image, (obj[1], obj[0]), (obj[3], obj[2]), color=(0, 0, 0), thickness=1)
#     cv2.imshow("slika", image)
#     cv2.waitKey(0)

import os
import cv2
import numpy as np
slika=r'E:\Science\Monika1\tamara\filtered_photos\test'
anot=r'E:\Science\Monika1\tamara\filtered_detections'

im_names=os.listdir(slika)


for im_name in im_names:
    # print(im_name)
    # print([im_name[:-4]])
    objects = []
    image = cv2.imread(os.path.join(slika, im_name))
    print(im_name)
    rows,col=image.shape[:2]
    print(rows, col)
    new_r=int(rows/2)
    new_c=int(col/2)
    image_res=cv2.resize(image,(new_c, new_r), interpolation=cv2.INTER_AREA)
    # image_res=cv2.reshape()
    for annot_name in (os.listdir(anot)):
        if (im_name[:-4] + '_') in annot_name or (im_name[:-4] + '.txt') in annot_name:
            print(image_res.shape)
            object = np.loadtxt(os.path.join(anot, annot_name), delimiter=',')

            object = [int(el) for el in object]
            pom=object[0]
            object[0]=object[1]
            object[1]=pom

            pom=object[2]
            object[2]=object[3]
            object[3]=pom
            #  print(object)
            # object[2]=object[0]+object[3]
            # object[3]=object[1]+object[2]
            # object[1] = object[1]
            # object[3] = object[3]
            # object = [int(np.round(object[0] )), int(np.round(object[1])), int(np.round(object[2])), int(np.round(object[3]))]

            objects.append(object)
            if len(objects)<0:
                continue
            print(object)
    for obj in objects:
        cv2.rectangle(image_res, (obj[1], obj[0]), (obj[3], obj[2]), color=(0, 255, 0), thickness=1)
        # print("crtam")
    cv2.imshow("slika", image_res)
    cv2.waitKey(0)
