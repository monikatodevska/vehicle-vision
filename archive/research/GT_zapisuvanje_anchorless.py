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
import shutil

import xml.etree.ElementTree as ET
import helper_model, helper_stats, helper_losses
import helper_data
import helper_anchorless_mc

version = 'vinf_12'
GroundTruthFilesTrain=r'E:\Science\Monika4\GroundTruthFilesAnchorless555'
# GroundTruthFilesVal=r'E:\Science\Monika\GroundTruthFilesVal'
srcImagesPath = r'E:\Science\Monika4'
# srcAnnotationsPath = r'E:\Science\Monika\GRAM-RTMv4\Annotations\site2'
srcAnnotationsPathTrain = r'E:\Science\Monika4\baseline_blurred'
dstResultsPath = r'E:\Science\Monika\Results'
dstModelsPath = r'E:\Science\Monika\Models'
# gtDstPath = r'E:\Science\Monika\GT'
annot_path_tamara=r'E:\Science\Monika4\anotacii'
# annot_path_tanja=r'E:\Science\Monika1\Novi\D2_train\anotacii'
# annot_video2=r'E:\Science\Monika4\anotacii'
# dst=r'E:\Science\Monika4\empty'
# augmented_dst=r'E:\Science\Monika4\augmented'
# if not os.path.exists(os.path.join(dstModelsPath, version)):
#         os.mkdir(os.path.join(dstModelsPath, version))
# modelsPath = os.path.join(dstModelsPath, version)
if not os.path.exists(GroundTruthFilesTrain):
         os.mkdir(GroundTruthFilesTrain)
# modelsPath = os.path.join(dstModelsPath, version)
imgDims = {'rows': 341, 'cols': 512}
num_classes = 2
img_depth = 1

img_dims = (imgDims['rows'], imgDims['cols'], img_depth)

im_size=(img_dims[1], img_dims[0])

# anchor_dims = ((18,18),(23, 23), (32, 32), (45, 45), (65, 65))
anchor_stride = 8
# norm_coef = 100  # constant to normalize regression ground truth data

# IOU thresholds for selecting positive and negative anchors
iou_low = 0.4
iou_high = 0.6
# num_cores = multiprocessing.cpu_count()
kernel = np.array([
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1]
    ]) / 9
def read_data_and_generate_gt(im_path, im_size, im_depth, annot_path, annot_path_tamara, annot_video2, GroundTruthFiles,m):
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
    for im_name in os.listdir(os.path.join(im_path)):
        # images_list = []  # array of normalized images
        object_annotations_list = []  # array of array of bounding boxes for each image
        # WScale = 800 / 1200
        # HScale = 480 / 720
        # valid_ind=[]
        # list images in source folder

        # flag1 = 0
        # print(im_name)
        # cv2.waitKey(0)
        # --- load image ---
        # print(im_name)
        # if not im_name[-4:] != '.bmp':  # exclude system files
        #     continue


        image = cv2.imread(os.path.join(im_path, im_name))

        rows,col=image.shape[:2]
        # print(col)
        # WScale = 512 / col
        # # print(WScale)
        # HScale = 341 / rows
        # print(im_name)
        # --- load annotations ---
        if im_name[0:5]=='image':

            # if not im_name[-4:] != '.bmp':  # exclude system files
            #     continue


            image = cv2.imread(os.path.join(im_path, im_name),0)

            rows, col = image.shape[:2]
            if im_size != (col, rows):
                image = cv2.resize(image, im_size, interpolation=cv2.INTER_AREA)
                # flag1 = 1
            image = image.reshape(image.shape[0], image.shape[1], im_depth)
            # print(1)
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
                if cl == 'car' and bb[3] - bb[2] + 1 > 15:
                    annot = [int(np.round(bb[2])), int(np.round(bb[0])), int(np.round(bb[3])), int(np.round(bb[1])), 1]
                    objects.append(annot)
                elif cl == "van" and bb[3] - bb[2] + 1 > 15:
                    annot = [int(np.round(bb[2])), int(np.round(bb[0])), int(np.round(bb[3])), int(np.round(bb[1])), 1]
                    objects.append(annot)
                elif cl == "truck" and bb[3] - bb[2] + 1 > 15:
                    annot = [int(np.round(bb[2])), int(np.round(bb[0])), int(np.round(bb[3])), int(np.round(bb[1])), 2]
                    objects.append(annot)
                else:
                    continue
            #     cv2.rectangle(image, (annot[1], annot[0]), (annot[3], annot[2]), color=(0, 0, 0), thickness=1)
            #
            # cv2.imshow("slika", image)
            # cv2.waitKey(0)


            # if len(objects)<0:
            #     continue
# &NOTE: otkoga ke se popravat kamionite da se vrati ovoj del
        elif(im_name[0:5]=="video") or (im_name[0:6]=="encode"):
            continue
            #     src_image = cv2.imread(os.path.join(im_path, im_name))
            #     rows, col = src_image.shape[:2]
            #     # desno = 50
            #     # levo = 248
            #     cropped_image = src_image[0:rows, 248:col - 50]
            #     rows_cr, col_cr = cropped_image.shape[:2]
            #     resized = cv2.resize(cropped_image, (512, 341), interpolation=cv2.INTER_AREA)
            #
            #     # resulting_image = cv2.filter2D(src_image, -1, kernel)
            #     # cv2.imwrite(os.path.join(trainBlur, str(im_name[:-4]) + '_b' + '.jpg'), resulting_image)
            #
            #
            #     objects = []
            #     HScale = 341 / rows_cr
            #     WScale = 512 / col_cr
            #     for annot_name in (os.listdir(annot_path_tanja)):
            #         if (im_name[:-4] + '_') in annot_name or (im_name[:-4] + '.txt') in annot_name:
            #             object = np.loadtxt(os.path.join(annot_path_tanja, annot_name), delimiter=',')
            #
            #             object = [int(el) for el in object]
            #             object[2] = object[0] + object[2]
            #             object[3] = object[1] + object[3]
            #             object[1] = object[1] - 248
            #             object[3] = object[3] - 248
            #             object = [int(np.round(object[0] * HScale)), int(np.round(object[1] * WScale)), int(np.round(object[2] * HScale)), int(np.round(object[3] * WScale))]
            #             objects.append(object)
            #
            #         else:
            #             continue


                        # za debug
                # for obj in objects:
                #     cv2.rectangle(resized, (obj[1], obj[0]), (obj[3], obj[2]), color=(0, 0, 0), thickness=1)
                # cv2.imshow("slika", resized)
                # cv2.waitKey(0)

                # cv2.imwrite(os.path.join(dst, im_name), cropped_image)
                # povikaj gt da se presmeta
                # zapisi vo fajl so ime na slika
            # print(objects)
            #         cv2.rectangle(image, (annot[1], annot[0]), (annot[3], annot[2]), color=(0, 0, 0), thickness=1)
            # cv2.imshow("slikaaaaa", image)
            # cv2.waitKey(0)
        elif(col==840): #tamara

            src_image = cv2.imread(os.path.join(im_path, im_name))
            rows, col = src_image.shape[:2]
            # desno = 50
            # levo = 248
            im_res = cv2.resize(src_image, (int(col / 2), int(rows / 2)), interpolation=cv2.INTER_AREA)
            rows_r, col_r = im_res.shape[:2]
            cropped_image = im_res[0:rows_r, 47:col_r - 34]
            rows_cr, col_cr = cropped_image.shape[:2]
            resized = cv2.resize(cropped_image, (512, 341), interpolation=cv2.INTER_AREA)
            # cv2.imwrite()
            objects = []
            HScale = 341 / rows_cr
            WScale = 512 / col_cr
            for annot_name in (os.listdir(annot_path_tamara)):
                if (im_name[:-4] + '_') in annot_name or (im_name[:-4] + '.txt') in annot_name:
                    object = np.loadtxt(os.path.join(annot_path_tamara, annot_name), delimiter=',')

                    object = [int(el) for el in object]

                    pom = object[0]
                    object[0] = object[1]
                    object[1] = pom

                    pom = object[2]
                    object[2] = object[3]
                    object[3] = pom
                    object[1] = object[1] - 47
                    object[3] = object[3] - 47
                    object = [int(np.round(object[0] * HScale)), int(np.round(object[1] * WScale)), int(np.round(object[2] * HScale)), int(np.round(object[3] * WScale)), int(object[4])]
                    objects.append(object)
        elif(col==1344): #od nikolaj slikite 1344 na 760
            image=cv2.imread(os.path.join(im_path, im_name))
            rows, col = image.shape[:2]
            cropped_image = image[0:rows, 0:1141]
            rows_cr, col_cr = cropped_image.shape[:2]
            resized = cv2.resize(cropped_image, (512, 341), interpolation=cv2.INTER_AREA)

            # resulting_image = cv2.filter2D(src_image, -1, kernel)
            # cv2.imwrite(os.path.join(trainBlur, str(im_name[:-4]) + '_b' + '.jpg'), resulting_image)


            objects = []
            HScale = 341 / rows_cr
            WScale = 512 / col_cr

            annot_name=im_name[:-4]+'.txt'
            objs=np.loadtxt(os.path.join(annot_video2, annot_name),delimiter=',', ndmin=2).astype(np.int)
            for bbox in objs:
                if(bbox[4]==4):
                    object=[int(bbox[0]*HScale), int(bbox[1]*WScale), int((bbox[0]+bbox[2])*HScale), int((bbox[1]+bbox[3])*WScale), 2]
                elif(bbox[4]==3):
                    object=[int(bbox[0]*HScale), int(bbox[1]*WScale), int((bbox[0]+bbox[2])*HScale), int((bbox[1]+bbox[3])*WScale), 1]
                else:
                    object=[int(bbox[0]*HScale), int(bbox[1]*WScale), int((bbox[0]+bbox[2])*HScale), int((bbox[1]+bbox[3])*WScale), int(bbox[4])]
                objects.append(object)
        elif(col==960): #kineski


        #     for obj in objects:
        #         cv2.rectangle(resized, (obj[1], obj[0]), (obj[3], obj[2]), color=(0, 0, 0), thickness=1)
        # cv2.imshow("slika", resized)
        # cv2.waitKey(0)
        if len(objects) < 0:
            shutil.move(os.path.join(im_path, im_name), os.path.join(dst, im_name))
            continue
                    # print(object)
            # for obj in objects:
            #     cv2.rectangle(resized, (obj[1], obj[0]), (obj[3], obj[2]), color=(0, 255, 0), thickness=1)
                # print("crtam")
            # cv2.imshow("slika", resized)
            # cv2.imshow("slika1", im_res)
            # cv2.waitKey(0)
        bboxes_train=deepcopy(objects)
        # out_class = []
        num_negs_ratio = 3
        obj_masks_train = helper_anchorless_mc.generate_anchor_level_object_masks(bboxes_train, img_dims, anchor_stride)
        out_class, out_reg=helper_anchorless_mc.get_anchorless_ground_truth_data_parallel_optimized(bboxes_train, obj_masks_train[:, :], img_dims, anchor_stride, iou_low, iou_high, num_negs_ratio, num_classes)

        if out_class is None and out_reg is None:
            shutil.move(os.path.join(im_path, im_name), os.path.join(dst, im_name))
            continue
        else:

            out_c_rows, out_c_cols, depth=out_class.shape
            out_r_rows, out_r_cols, depth_r=out_reg.shape
            out_class_dims=[out_c_rows, out_c_cols, depth]
            out_reg_dims=[out_r_rows, out_r_cols, depth_r]
            out_class_flat=np.ndarray.flatten(out_class)

            out_reg_flat=np.ndarray.flatten(out_reg)


            # filename=str(im_name[:-4]) +'.txt'
            # fid=open(os.path.join(GroundTruthFiles, filename), 'wb+')
            # pickle.dump(out_class_dims,fid)
            # pickle.dump(out_class_flat, fid)
            # pickle.dump(out_reg_dims,fid)
            # pickle.dump(out_reg_flat,fid)
            # fid.close()
            if(im_name[0:5]=="image"):

                filename=str(im_name[:-4]) +'.txt'
                fid=open(os.path.join(GroundTruthFiles, filename), 'wb+')
                pickle.dump(out_class_dims,fid)
                pickle.dump(out_class_flat, fid)
                pickle.dump(out_reg_dims,fid)
                pickle.dump(out_reg_flat,fid)
                fid.close()

                im_name_b=int(im_name[5:11])
                im_name_b_1=im_name_b+16910
                im_name_new='image' + str(im_name_b_1).zfill(6)
                filename = str(im_name_new) + '.txt'
                fid_b = open(os.path.join(GroundTruthFiles, filename), 'wb+')
                pickle.dump(out_class_dims, fid_b)
                pickle.dump(out_class_flat, fid_b)
                pickle.dump(out_reg_dims, fid_b)
                pickle.dump(out_reg_flat, fid_b)
                fid_b.close()


                im_name_f=im_name_b_1+16910
                im_name_new_f = 'image' + str(im_name_f).zfill(6)
                filename = str(im_name_new_f) + '.txt'
                fid_f = open(os.path.join(GroundTruthFiles, filename), 'wb+')

                out_class_f=np.fliplr(out_class)
                out_class_flat_f=np.ndarray.flatten(out_class_f)

                out_reg_f=np.fliplr(out_reg)
                out_reg_flip_f=np.ndarray.flatten(out_reg_f)
                pickle.dump(out_class_dims, fid_f)
                pickle.dump(out_class_flat_f, fid_f)
                pickle.dump(out_reg_dims, fid_f)
                pickle.dump(out_reg_flip_f, fid_f)
                fid_f.close()

                im_name_f1 = im_name_f + 16910
                im_name_new_f1 = 'image' + str(im_name_f1).zfill(6)
                filename = str(im_name_new_f1) + '.txt'
                fid_f1 = open(os.path.join(GroundTruthFiles, filename), 'wb+')
                # out_class_f = np.fliplr(out_class)
                pickle.dump(out_class_dims, fid_f1)
                pickle.dump(out_class_flat_f, fid_f1)
                pickle.dump(out_reg_dims, fid_f1)
                pickle.dump(out_reg_flip_f, fid_f1)
                fid_f1.close()
            else:
                #obicno zapisi slika i gt (renaming)
                im_name_new='image'+str(m).zfill(6) +'.jpg'
                cv2.imwrite(os.path.join(augmented_dst, im_name_new), resized)

                filename ='image'+str(m).zfill(6) + '.txt'

                fid = open(os.path.join(GroundTruthFiles, filename), 'wb+')
                pickle.dump(out_class_dims, fid)
                pickle.dump(out_class_flat, fid)
                pickle.dump(out_reg_dims, fid)
                pickle.dump(out_reg_flat, fid)
                fid.close()
                m+=1


                #blur
                im_name_new='image'+str(m).zfill(6) +'.jpg'
                resulting_image = cv2.filter2D(resized, -1, kernel)
                cv2.imwrite(os.path.join(augmented_dst,im_name_new ), resulting_image)

                gt_blurred_filename='image'+str(m).zfill(6) + '.txt'
                fid_b = open(os.path.join(GroundTruthFiles, gt_blurred_filename), 'wb+')
                pickle.dump(out_class_dims, fid_b)
                pickle.dump(out_class_flat, fid_b)
                pickle.dump(out_reg_dims, fid_b)
                pickle.dump(out_reg_flat, fid_b)
                fid_b.close()
                m+=1

                #regular flip
                im_name_new='image'+str(m).zfill(6) +'.jpg'
                regular_flipped = cv2.flip(resized, 1)
                cv2.imwrite(os.path.join(augmented_dst, im_name_new), regular_flipped)

                gt_flipped_filename='image'+str(m).zfill(6) + '.txt'
                fid_f = open(os.path.join(GroundTruthFiles, gt_flipped_filename), 'wb+')
                # flip gt

                out_class_f = np.fliplr(out_class)
                out_class_flat_f = np.ndarray.flatten(out_class_f)

                out_reg_f = np.fliplr(out_reg)
                out_reg_flip_f = np.ndarray.flatten(out_reg_f)

                pickle.dump(out_class_dims, fid_f)
                pickle.dump(out_class_flat_f, fid_f)
                pickle.dump(out_reg_dims, fid_f)
                pickle.dump(out_reg_flip_f, fid_f)
                fid_f.close()
                m+=1

                #blurred flip
                im_name_new='image'+str(m).zfill(6) +'.jpg'
                blured_flipped = cv2.flip(resulting_image, 1)
                # name = int(im_name[5:11]) + k
                # im_name1 = 'image' + str(name).zfill(6) + '.jpg'
                cv2.imwrite(os.path.join(augmented_dst,im_name_new ), blured_flipped)

                gt_flipped_blurred_filename = 'image'+str(m).zfill(6) + '.txt'
                fid_f1 = open(os.path.join(GroundTruthFiles, gt_flipped_blurred_filename), 'wb+')
                # out_class_f = np.fliplr(out_class)
                pickle.dump(out_class_dims, fid_f1)
                pickle.dump(out_class_flat_f, fid_f1)
                pickle.dump(out_reg_dims, fid_f1)
                pickle.dump(out_reg_flip_f, fid_f1)
                fid_f1.close()
                m+=1
                # gt_blurred_filename=im_name[:-4]+'_b'+'.txt'
                # fid_b = open(os.path.join(GroundTruthFiles, gt_blurred_filename), 'wb+')
                # pickle.dump(out_class_dims, fid_b)
                # pickle.dump(out_class_flat, fid_b)
                # pickle.dump(out_reg_dims, fid_b)
                # pickle.dump(out_reg_flat, fid_b)
                # fid_b.close()

                # gt_flipped_filename=im_name[:-4]+'_f'+'.txt'
                # fid_f = open(os.path.join(GroundTruthFiles, gt_flipped_filename), 'wb+')
                # #flip gt
                #
                # out_class_f = np.fliplr(out_class)
                # out_class_flat_f = np.ndarray.flatten(out_class_f)
                #
                # out_reg_f = np.fliplr(out_reg)
                # out_reg_flip_f = np.ndarray.flatten(out_reg_f)
                #
                # pickle.dump(out_class_dims, fid_f)
                # pickle.dump(out_class_flat_f, fid_f)
                # pickle.dump(out_reg_dims, fid_f)
                # pickle.dump(out_reg_flip_f, fid_f)
                # fid_f.close()

                # gt_flipped_blurred_filename=im_name[:-4]+'_f_b'+'.txt'
                # fid_f1 = open(os.path.join(GroundTruthFiles,gt_flipped_blurred_filename), 'wb+')
                # # out_class_f = np.fliplr(out_class)
                # pickle.dump(out_class_dims, fid_f1)
                # pickle.dump(out_class_flat_f, fid_f1)
                # pickle.dump(out_reg_dims, fid_f1)
                # pickle.dump(out_reg_flip_f, fid_f1)
                # fid_f1.close()





            # fid1=open(os.path.join(GroundTruthFiles, filename), 'rb')
            # out_class_back=pickle.load(fid1)
            # out_class_back1=pickle.load(fid1)
            #
            # out_class_back1=np.reshape(out_class_back,(out_c_rows,out_c_cols,depth))


# result_val=Parallel(n_jobs=-1, temp_folder="/tmp", max_nbytes=None, backend="multiprocessing")(delayed(read_data_and_generate_gt)(im_name, os.path.join(srcImagesPath, 'site-baseline-cropped-resized'),(imgDims['cols'], imgDims['rows']),img_depth,srcAnnotationsPathTrain, GroundTruthFilesTrain) for im_name in os.listdir(os.path.join(srcImagesPath,'site-baseline-cropped-resized')))

m=801
srcAnnotationsPathVal=r'E:\Science\Monika4\video4_anotacii'
GroundTruthFilesVal=r'E:\Science\Monika4\GT_val'
annot_video2=r'E:\Science\Monika4\video4_anotacii'
dst=r'E:\Science\Monika4\empty_val'
augmented_dst=r'E:\Science\Monika4\augmented_val'
# read_data_and_generate_gt(os.path.join(srcImagesPath, 'za-gt'),(imgDims['cols'], imgDims['rows']),img_depth,srcAnnotationsPathTrain, annot_path_tamara, annot_video2, GroundTruthFilesTrain,m)
read_data_and_generate_gt(os.path.join(srcImagesPath, 'video4_val'),(imgDims['cols'], imgDims['rows']),img_depth,srcAnnotationsPathVal,annot_path_tamara,annot_video2, GroundTruthFilesVal,m)

