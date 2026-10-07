import os
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from joblib import Parallel, delayed
import multiprocessing
import pickle
import h5py
# custom package imports
#from SingleShotDetector.Helpers import helper_model, helper_data, helper_stats, losses
import sys
#sys.path.insert(0, 'E:\Science\Monika\SingleShotDetector\Helpers' )
import cv2
import random
#from tqdm import tqdm

import xml.etree.ElementTree as ET
import helper_model, helper_stats, helper_losses
import helper_data


version = 'vinf_12'
GroundTruthFilesTrain=r'E:\Science\Monika\GroundTruthFiles'
# GroundTruthFilesVal=r'E:\Science\Monika\GroundTruthFilesVal'
srcImagesPath = r'E:\Science\Monika1'
# srcAnnotationsPath = r'E:\Science\Monika\GRAM-RTMv4\Annotations\site2'
srcAnnotationsPathTrain = r'E:\Science\Monika\GRAM-RTMv4\Annotations\baseline_blurred'
dstResultsPath = r'E:\Science\Monika\Results'
dstModelsPath = r'E:\Science\Monika\Models'
gtDstPath = r'E:\Science\Monika\GT'

if not os.path.exists(os.path.join(dstModelsPath, version)):
        os.mkdir(os.path.join(dstModelsPath, version))
modelsPath = os.path.join(dstModelsPath, version)

imgDims = {'rows': 341, 'cols': 512}
num_classes = 1
img_depth = 1

img_dims = (imgDims['rows'], imgDims['cols'], img_depth)

im_size=(img_dims[1], img_dims[0])

anchor_dims = ((32,32), (48,48), (64,64), (92,92))
anchor_stride = 8
norm_coef = 100  # constant to normalize regression ground truth data

# IOU thresholds for selecting positive and negative anchors
iou_low = 0.4
iou_high = 0.6
# num_cores = multiprocessing.cpu_count()

def read_data_and_generate_gt(im_path, im_size, im_depth, annot_path, GroundTruthFiles):
    """
    load, resize, and normalize image data
    loads images and annotations from one folder
    :param im_path: global path of folder containing images of a data subset [string]
    :param im_size: output dimensions of the images (cols, rows) [tuple]
    :param im_depth: required depth of the loaded images (value: 1 or 3) [int]
    :param shuffle: whether to shuffle input data order [bool]
    :return: images_list - array of normalized depth maps [ndarray]
             object_annotations_list - annotated bounding boxes min_row, min_col, max_row, max_col [list]
    """

    # images_list = []  # array of normalized images
    object_annotations_list = []  # array of array of bounding boxes for each image
    # WScale = 800 / 1200
    # HScale = 480 / 720
    # valid_ind=[]
    # list images in source folder
    for im_ind, im_name in enumerate(os.listdir(im_path)):
        # flag1 = 0
        # print(im_name)
        # cv2.waitKey(0)
        # --- load image ---

        if not im_name[-4:] != '.bmp':  # exclude system files
            continue

        if im_depth == 3:
            image = cv2.imread(os.path.join(im_path, im_name))
        else:
            image = cv2.imread(os.path.join(im_path, im_name), 0)
        # rows, col = image.shape[:2]
        # if im_size != (col, rows):
        #     image = cv2.resize(image, im_size, interpolation=cv2.INTER_AREA)
            # flag1 = 1
        image = image.reshape(image.shape[0], image.shape[1], im_depth)
        # WScale = 512 / col
        # print(WScale)
        # HScale = 341 / rows
        # print(im_name)
        # --- load annotations ---
        annot_name = str(int(im_name[5:11].lstrip('0')) - 1) + '.xml'
        # print(annot_name)

        root = ET.parse(os.path.join(annot_path, annot_name)).getroot()

        objects = []  # list of all objects in the image
        # cv2.imwrite(os.path.join(im_path_resized, im_name), image)
        for object in root.findall('object'):

            cl = object.find('class').text

            bb_xml = object.find('bndbox')
            bb = [np.int(bb_xml.find('xmin').text),  # min_col
                  np.int(bb_xml.find('xmax').text),  # max_col
                  np.int(bb_xml.find('ymin').text),  # min_row
                  np.int(bb_xml.find('ymax').text),  # max_row
                  ]
            # if flag1 == 1:
                # print(1)
            annot = [int(np.round(bb[2])), int(np.round(bb[0])), int(np.round(bb[3])), int(np.round(bb[1]))]

            # annot_n=[bb[2], bb[0], bb[3], bb[1]]
            # min_row, min_col, max_row, max_col
            # select positive car samples, height > 25px
            if cl == 'car' and annot[2] - annot[0] + 1 > 20:
                # print("zemen")
                objects.append(annot)
            # if im_ind==121:
            #     print(bb[2], bb[0], bb[3], bb[1])
            #     print(annot)
            #     cv2.rectangle(image, (annot[1], annot[0]), (annot[3], annot[2]), color=(0, 0, 0), thickness=1)

            #     cv2.imshow("k", image)
            #     cv2.waitKey(0)
        if len(objects)<0:
            continue


        # print(objects)
        #         cv2.rectangle(image, (annot[1], annot[0]), (annot[3], annot[2]), color=(0, 0, 0), thickness=1)
        # cv2.imshow("slikaaaaa", image)
        # cv2.waitKey(0)
            # else:
            #     annot = [bb[2], bb[0], bb[3], bb[1]]
            #     if cl == 'car' and bb[3] - bb[2] + 1 > 20:
                    # cv2.rectangle(imgs[img_ind], (bbox[1], bbox[0]), (bbox[3], bbox[2]), color=(0, 0, 0), thickness=1)
                    # cv2.imshow("slika", imgs[img_ind])
                    # cv2.waitKey(0)
                    # objects.append(annot)
            # cv2.rectangle(image, (annot[1], annot[0]), (annot[3], annot[2]), color=(0, 0, 0), thickness=1)
        # cv2.rectangle(image, (annot_n[1], annot_n[0]), (annot_n[3], annot_n[2]), color=(255, 0, 0), thickness=1)
        #  cv2.imshow("slika", image)
        #  cv2.waitKey(0)
        bboxes_train=deepcopy(objects)

        out_class=helper_data.get_anchor_data_ssd(bboxes_train, anchor_dims, img_dims, anchor_stride, iou_low, iou_high)
        if out_class is None:
            continue
        else:
            out_c_rows, out_c_cols, depth=out_class.shape
            out_class_dims=[out_c_rows, out_c_cols, depth]
            out_class_flat=np.ndarray.flatten(out_class)
            filename=str(im_name) +'_' + str(iou_low) + '_' + str(iou_high) +'_' +str(anchor_stride)+'.txt'
            fid=open(os.path.join(GroundTruthFiles, filename), 'wb+')
            pickle.dump(out_class_dims,fid)
            pickle.dump(out_class_flat, fid)
            fid.close()

            im_name_b=int(im_name[5:11])
            im_name_b_1=im_name_b+16910
            im_name_new='image' + str(im_name_b_1).zfill(6)+'.jpg'
            filename = str(im_name_new) + '_' + '.txt'
            fid_b = open(os.path.join(GroundTruthFiles, filename), 'wb+')
            pickle.dump(out_class_dims, fid_b)
            pickle.dump(out_class_flat, fid_b)
            fid_b.close()


            im_name_f=im_name_b_1+16910
            im_name_new_f = 'image' + str(im_name_f).zfill(6) + '.jpg'
            filename = str(im_name_new_f) + '_' + '.txt'
            fid_f = open(os.path.join(GroundTruthFiles, filename), 'wb+')
            out_class_f=np.fliplr(out_class)
            pickle.dump(out_class_dims, fid_f)
            pickle.dump(out_class_f, fid_f)
            fid_f.close()

            im_name_f1 = im_name_f + 16910
            im_name_new_f1 = 'image' + str(im_name_f1).zfill(6) + '.jpg'
            filename = str(im_name_new_f1) + '_' + '.txt'
            fid_f1 = open(os.path.join(GroundTruthFiles, filename), 'wb+')
            # out_class_f = np.fliplr(out_class)
            pickle.dump(out_class_dims, fid_f1)
            pickle.dump(out_class_f, fid_f1)
            fid_f1.close()

            # fid1=open(os.path.join(GroundTruthFiles, filename), 'rb')
            # out_class_back=pickle.load(fid1)
            # out_class_back1=pickle.load(fid1)
            #
            # out_class_back1=np.reshape(out_class_back,(out_c_rows,out_c_cols,depth))




read_data_and_generate_gt(os.path.join(srcImagesPath, 'za-gt'),(imgDims['cols'], imgDims['rows']),img_depth,srcAnnotationsPathTrain, GroundTruthFilesTrain)
# read_data_and_generate_gt(os.path.join(srcImagesPath, 'val1'),(imgDims['cols'], imgDims['rows']),img_depth,srcAnnotationsPath, GroundTruthFilesVal)
