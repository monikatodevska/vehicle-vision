import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET

trainOriginal = r'E:\Science\Monika1\blurred_cropped_resized'
trainFlip=r'E:\Science\Monika1\flipped'
srcAnnotationsPathTrainCopy = r'E:\Science\Monika\GRAM-RTMv4\Annotations\flipped'
listOrig=os.listdir(trainOriginal)
srcAnnotationsPathTrainCopy1 = r'E:\Science\Monika\GRAM-RTMv4\Annotations\flipped1'
k=33820
for im_ind, im_name in enumerate(listOrig):
    src_image = cv2.imread(os.path.join(trainOriginal, im_name))
    rows, col = src_image.shape[:2]
    # print(col)
    resulting_image = cv2.flip(src_image,1)
    name=int(im_name[5:11])+k
    im_name1='image'+ str(name).zfill(6) + '.jpg'
    cv2.imwrite(os.path.join(trainFlip, im_name1 ), resulting_image)
    annot_name = str(int(im_name[5:11].lstrip('0')) - 1) + '.xml'
    xmlTree = ET.parse(os.path.join(srcAnnotationsPathTrainCopy, annot_name))
    root = xmlTree.getroot()

    objects = []  # list of all objects in the image
    for object in root.findall('object'):
        # cl = object.find('class').text

        bb_xml = object.find('bndbox')
        max_row = np.int(bb_xml.find('ymax').text)
        min_row = np.int(bb_xml.find('ymin').text)


        min_col=np.int(bb_xml.find('xmin').text)

        max_col=np.int(bb_xml.find('xmax').text)
        # print(min_col)
        # print(max_col)
        newmin_col=col-min_col-1
        newmax_col=col-max_col-1
        # print(newmin_col)
        bb_xml.find('xmin').text=str(newmax_col)
        # print(bb_xml.find('xmin').text)
        bb_xml.find('xmax').text=str(newmin_col)
        # print(newmax_col)
        # if np.int(bb_xml.find('xmax').text)< np.int(bb_xml.find('xmin').text):
        #     cv2.rectangle(resulting_image, (newmin_col, min_row), (newmax_col, max_row), color=(0, 0, 0), thickness=1)
        #     cv2.imshow('slika', resulting_image)
        #     cv2.waitKey(0)
    annot_name1=name-1
    annot_name2=str(annot_name1)+'.xml'
    xmlTree.write(os.path.join(srcAnnotationsPathTrainCopy1, annot_name2))
    # print(1)

# filename='0.xml'
# srcAnnotationsPathTrainCopy1 = r'E:\Science\Monika\GRAM-RTMv4\Annotations\train-annot-copy1'
# #
# impath=r'E:\Science\Monika\train-flipped'
# imagenames=['image047348.jpg', 'image023674.jpg']
# filenames='47347.xml'
# filename1='47347_1.xml'
# for img_ind, imagename in enumerate(imagenames):
#     image=cv2.imread(os.path.join(impath,imagenames[img_ind]))
#     cv2.imshow('sl', image)
#     annot_name = str(int(imagename[5:11].lstrip('0')) - 1) + '.xml'
#     xmlTree=ET.parse(os.path.join(srcAnnotationsPathTrainCopy, annot_name))
#     root=xmlTree.getroot()
#
#     for object in root.findall('object'):
#         bb_xml = object.find('bndbox')
#         min_row = np.int(bb_xml.find('ymin').text)
#         min_col = np.int(bb_xml.find('xmin').text)
#         max_row = np.int(bb_xml.find('ymax').text)
#         max_col = np.int(bb_xml.find('xmax').text)
#
#         cv2.rectangle(image, (min_col, min_row), (max_col,max_row),color=(0, 0, 0), thickness=1)
#
#     cv2.imwrite(os.path.join(srcAnnotationsPathTrainCopy1,imagename), image)

