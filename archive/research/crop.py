Cimport os
import cv2
import math
import xml.etree.ElementTree as ET
import numpy as np
trainOriginal = r'E:\Science\Monika1\za-gt'
dst = r'E:\Science\Monika1\Novi\dodaj'
# srcAnnotationsPathTrainCopy1 = r'E:\Science\Monika\GRAM-RTMv4\Annotations\baseline_blurred'
# annot_path= r'E:\Science\Monika\GRAM-RTMv4\Annotations\M-30-anotacii'
listOrig=os.listdir(trainOriginal)
# anotacii=r'E:\Science\Monika1\Novi\D2_train\anotacii'
# listOrig=os.listdir(trainOriginal)
k=1
kernel=np.array([
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1]
]) / 9
for im_ind, im_name in enumerate(listOrig):
    src_image = cv2.imread(os.path.join(trainOriginal, im_name))
    rows, col = src_image.shape[:2]
    if (im_name[0:5] == "video") or (im_name[0:6] == "encode"):

        # desno = 50
        # levo = 248
        cropped_image = src_image[0:rows, 248:col - 50]

    elif (im_name[0:5]=="image"):
        continue

    else:
        im_res = cv2.resize(src_image, (int(col / 2), int(rows / 2)), interpolation=cv2.INTER_AREA)
        rows_r, col_r = im_res.shape[:2]
        cropped_image = im_res[0:rows_r, 47:col_r - 34]
        rows_cr, col_cr = cropped_image.shape[:2]
        # resized = cv2.resize(cropped_image, (512, 341), interpolation=cv2.INTER_AREA)


        # rows_cr, col_cr = cropped_image.shape[:2]
    resized = cv2.resize(cropped_image, (512, 341), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(dst, im_name), resized) #obicna  cropnata i resiznata

    #blur and write
    resulting_image = cv2.filter2D(resized, -1, kernel)
    cv2.imwrite(os.path.join(dst, str(im_name[:-4]) + '_b' + '.jpg'), resulting_image)

    regular_flipped = cv2.flip(resized, 1)
    blured_flipped = cv2.flip(resulting_image, 1)
    # name = int(im_name[5:11]) + k
    # im_name1 = 'image' + str(name).zfill(6) + '.jpg'
    cv2.imwrite(os.path.join(dst, im_name[:-4] + '_f' + '.jpg'), regular_flipped)
    cv2.imwrite(os.path.join(dst, im_name[:-4] + '_f_b' + '.jpg'), blured_flipped)


    # src_image = cv2.imread(os.path.join(trainOriginal, im_name))
    # rows, col = src_image.shape[:2]
    # desno=50
    # levo=248
    # cropped_image=src_image[0:rows, 248:col-50]
    # rows_cr, col_cr=cropped_image.shape[:2]
    # resized=cv2.resize(cropped_image,(512,341),interpolation=cv2.INTER_AREA)
    #
    # objects=[]
    # HScale=341/rows_cr
    # WScale=512/col_cr
    # for annot_name in (os.listdir(anotacii)):
    #     if(im_name[:-4]+'_') in annot_name or (im_name[:-4]+'.txt') in annot_name:
    #         object=np.loadtxt(os.path.join(anotacii,annot_name), delimiter=',')
    #
    #         object = [int(el) for el in object]
    #         object[2]=object[0]+object[2]
    #         object[3]=object[1]+object[3]
    #         object[1]=object[1]-248
    #         object[3]=object[3]-248
    #         object = [int(np.round(object[0] * HScale)), int(np.round(object[1] * WScale)), int(np.round(object[2] * HScale)), int(np.round(object[3] * WScale))]
    #         objects.append(object)
        # elif(im_name[:-4]+'.txt') in annot_name:
        #     object = np.loadtxt(os.path.join(anotacii,annot_name), delimiter=',')
        #     object = [int(el) for el in object]
        #     object[2] = object[0] + object[2]
        #     object[3] = object[1] + object[3]
        #     object[1] = object[1] - 248
        #     object[3] = object[3] - 248
        #     object = [int(np.round(object[0] * HScale)), int(np.round(object[1] * WScale)), int(np.round(object[2] * HScale)), int(np.round(object[3] * WScale))]
        #     objects.append(object)
        # else:
        #     continue
            #za debug
    # for obj in objects:
    #     cv2.rectangle(resized, (obj[1], obj[0]), (obj[3], obj[2]), color=(0, 0, 0), thickness=1)
    # cv2.imshow("slika", resized)
    # cv2.waitKey(0)











    # cv2.imwrite(os.path.join(dst, im_name), cropped_image)
    #povikaj gt da se presmeta
    #zapisi vo fajl so ime na slika



# for im_ind, im_name in enumerate(listOrig):
#     src_image = cv2.imread(os.path.join(trainOriginal, im_name))
#     rows, col = src_image.shape[:2]
#     id=int(im_name[5:11])
#     # WScale=512/col
#     # HScale=341/rows
#     # print(WScale)
#     # WScale1 = 512 / 800
#     # HScale1= 341 / 480
#     if(id<=7520) or (id>=16911 and id<=24430):
#         # print(col)
#         k=int(col-70)
#         m=int((col-70)*(rows/col))
#         gore=rows-m
#         cropped_image=src_image[gore:rows, 0:k]
#         rows,col=cropped_image.shape[:2]
#         # print(col)
#         # print(rows)
#         WScale=512/col
#         HScale=341/rows
#         resized=cv2.resize(cropped_image,(512,341),interpolation=cv2.INTER_AREA)
#         annot_name = str(int(im_name[5:11].lstrip('0')) - 1) + '.xml'
#         # print(annot_name)
#         xmlTree = ET.parse(os.path.join(annot_path, annot_name))
#         root = xmlTree.getroot()
#         # root = ET.parse(os.path.join(annot_path, annot_name)).getroot()
#
#         objects = []  # list of all objects in the image
#         # cv2.imwrite(os.path.join(im_path_resized, im_name), image)
#         for object in root.findall('object'):
#             cl = object.find('class').text
#
#             bb_xml = object.find('bndbox')
#             bb = [np.int(bb_xml.find('xmin').text),  # min_col
#                   np.int(bb_xml.find('xmax').text),  # max_col
#                   np.int(bb_xml.find('ymin').text),  # min_row
#                   np.int(bb_xml.find('ymax').text),  # max_row
#                   ]
#             # if flag1 == 1:
#             # print( max(0,int(bb[1])))
#             annot=[int(bb[2])-gore, max(0,int(bb[0])),int(bb[3])-gore, max(0,int(bb[1]))]
#             # print(annot)
#             annot = [int(np.round(annot[0] * HScale)), int(np.round(annot[1] * WScale)), int(np.round(annot[2] * HScale)), int(np.round(annot[3] * WScale))]
#             bb_xml.find('ymax').text=str(annot[2])
#             bb_xml.find('ymin').text=str(annot[0])
#
#             bb_xml.find('xmin').text=str(annot[1])
#
#             bb_xml.find('xmax').text=str(annot[3])
#
#             # print(annot)
#             # cv2.rectangle(resized, (annot[1], annot[0]), (annot[3], annot[2]), color=(0, 0, 0), thickness=1)
#             # cv2.rectangle(resized, (annot1[1], annot1[0]), (annot1[3], annot1[2]), color=(0, 0, 255), thickness=1)
#
#
#         # cv2.imshow(os.path.join(trainOriginal, im_name), resized)
#         # cv2.waitKey(0)
#         cv2.imwrite(os.path.join(dst, im_name),resized)
#         xmlTree.write(os.path.join(srcAnnotationsPathTrainCopy1, annot_name))
#
#     elif (id > 7520 and id<=16910) or (id>=24431):
#         # WScale1 = 512 / col
#         # HScale1 = 341 / rows
#         # resized2 = cv2.resize(src_image, (512, 341), interpolation=cv2.INTER_AREA)
#
#         k=119
#         m=int((col-k)*(rows/col))
#         gore=rows-m
#         cropped_image=src_image[gore:rows, k+1:col]
#         rows,col=cropped_image.shape[:2]
#         # print(rows)
#         # print(col)
#
#         WScale = 512 / col
#         HScale = 341 / rows
#         resized = cv2.resize(cropped_image, (512, 341), interpolation=cv2.INTER_AREA)
#         annot_name = str(int(im_name[5:11].lstrip('0')) - 1) + '.xml'
#         # print(annot_name)
#         xmlTree = ET.parse(os.path.join(annot_path, annot_name))
#         root = xmlTree.getroot()
#         objects = []
#
#         for object in root.findall('object'):
#             cl = object.find('class').text
#
#             bb_xml = object.find('bndbox')
#             bb = [np.int(bb_xml.find('xmin').text),  # min_col
#                   np.int(bb_xml.find('xmax').text),  # max_col
#                   np.int(bb_xml.find('ymin').text),  # min_row
#                   np.int(bb_xml.find('ymax').text),  # max_row
#                   ]
#             # if flag1 == 1:
#             # print(1)
#             annot=[int(bb[2])-gore, min(col,int(bb[0])-k),int(bb[3])-gore, min(col,int(bb[1])-k)]
#             # annot1=[int(bb[2]),int(bb[0]),int(bb[3]), int(bb[1])]
#             # print(annot)
#             annot = [int(np.round(annot[0] * HScale)), int(np.round(annot[1] * WScale)), int(np.round(annot[2] * HScale)), int(np.round(annot[3] * WScale))]
#             # annot2 = [int(np.round(annot1[0] * HScale1)), int(np.round(annot1[1] * WScale1)), int(np.round(annot1[2] * HScale1)), int(np.round(annot1[3] * WScale1))]
#
#             # cv2.rectangle(src_image, (annot1[1], annot1[0]), (annot1[3], annot1[2]), color=(0, 0, 0), thickness=1)
#             # cv2.rectangle(resized, (annot[1], annot[0]), (annot[3], annot[2]), color=(0, 0, 255), thickness=1)
#             # cv2.rectangle(resized2, (annot2[1], annot2[0]), (annot2[3], annot2[2]), color=(0, 255, 255), thickness=1)
#
#         # cv2.imshow(os.path.join(trainOriginal, im_name), src_image)
#         #
#         # cv2.imshow("slika2", resized)
#         # # cv2.imshow("slika3", resized2)
#         #
#         # cv2.waitKey(0)
#             bb_xml.find('ymax').text=str(annot[2])
#             bb_xml.find('ymin').text=str(annot[0])
#
#             bb_xml.find('xmin').text=str(annot[1])
#
#             bb_xml.find('xmax').text=str(annot[3])
#
#         cv2.imwrite(os.path.join(dst, im_name), resized)
#         xmlTree.write(os.path.join(srcAnnotationsPathTrainCopy1, annot_name))