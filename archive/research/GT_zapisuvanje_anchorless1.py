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
# import anchorless_genertor_plot
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
# import anchorless_genertor_plot
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
# GroundTruthFilesTrain=r'E:\Science\Monika4\GroundTruthFilesAnchorless555'
# GroundTruthFilesVal=r'E:\Science\Monika\GroundTruthFilesVal'
# srcImagesPath = r'E:\Science\Monika4'
# srcAnnotationsPath = r'E:\Science\Monika\GRAM-RTMv4\Annotations\site2'
srcAnnotationsPathTrain = r'E:\Science\Monika4\baseline_blurred'
dstResultsPath = r'E:\Science\Monika\Results'
dstModelsPath = r'E:\Science\Monika\Models'
# gtDstPath = r'E:\Science\Monika\GT'
# annot_path_tamara=r'E:\Science\Monika4\anotacii'
# annot_path_tanja=r'E:\Science\Monika1\Novi\D2_train\anotacii'
# annot_video2=r'E:\Science\Monika4\anotacii'
# dst=r'E:\Science\Monika4\empty'
# augmented_dst=r'E:\Science\Monika4\augmented'
# if not os.path.exists(os.path.join(dstModelsPath, version)):
#         os.mkdir(os.path.join(dstModelsPath, version))
# modelsPath = os.path.join(dstModelsPath, version)
res=r'D:\KlasifikacijaVozila\Miladinovci\crtanje'

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
iou_low = 0.6
iou_high = 0.7
# num_cores = multiprocessing.cpu_coun t()
kernel = np.array([
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1]
    ]) / 9
def read_data_and_generate_gt(im_root, annot_root, im_size, im_depth, GroundTruthFiles,augmented_dst, m):
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
    nemapoz=0
    nemaanot=0
    prazenfajl=0
    frli=0
    broj=0
    dirnames=os.listdir(im_root)
    print(dirnames)
    for dir_name in dirnames:
        # print(dir_name)
        im_path = os.path.join(im_root, dir_name)
        annot_path = os.path.join(annot_root, dir_name)
        # print (im_path)
        filenames=os.listdir(im_path)
        # print(filenames)
        for im_name in filenames:
            flag=0
            # if im_name[-4:] != '.bmp' or im_name[-4:] != '.jpg':
            #     continue
            if im_name[-3:]=='.db':
                continue
            if 'videoframes' in dir_name:
                break
                # print('videoframes')
                image = cv2.imread(os.path.join(im_path, im_name))
                rows, col = image.shape[:2]
                cropped_image = image[0:rows, 0:1141]
                rows_cr, col_cr = cropped_image.shape[:2]
                resized = cv2.resize(cropped_image, (512, 341), interpolation=cv2.INTER_AREA)

                # resulting_image = cv2.filter2D(src_image, -1, kernel)
                # cv2.imwrite(os.path.join(trainBlur, str(im_name[:-4]) + '_b' + '.jpg'), resulting_image)

                objects = []
                HScale = 341 / rows_cr
                WScale = 512 / col_cr

                annot_name = im_name[:-4] + '.txt'
                if os.path.exists(os.path.join(annot_path, annot_name)):
                    if os.stat(os.path.join(annot_path, annot_name)).st_size != 0:
                        # print(annot_name)
                        # print(annot_path)
                        objs = np.loadtxt(os.path.join(annot_path, annot_name), delimiter=',', ndmin=2).astype(np.int)
                        for bbox in objs:
                            if (bbox[4] == 4):
                                object = [int((bbox[0]-1) * HScale), int((bbox[1]-1) * WScale), int((bbox[0]-1 + bbox[2]) * HScale), int((bbox[1]-1 + bbox[3]) * WScale), 2]
                            elif (bbox[4] == 3):
                                object = [int((bbox[0]-1) * HScale), int((bbox[1]-1) * WScale), int((bbox[0]-1 + bbox[2]) * HScale), int((bbox[1]-1 + bbox[3]) * WScale), 1]
                            else:
                                object = [int((bbox[0]-1) * HScale), int((bbox[1]-1) * WScale), int((bbox[0]-1 + bbox[2]) * HScale), int((bbox[1]-1 + bbox[3]) * WScale), int(bbox[4])]
                            objects.append(object)
                        for obj in objects:
                            cv2.rectangle(resized, (obj[1], obj[0]), (obj[3], obj[2]), color=(0, 255, 0), thickness=1)
                            # print("crtam")
                        # cv2.imshow("slika", resized)
                        # cv2.imshow("slika1", im_res)
                        # cv2.waitKey(0)
                    else:
                        prazenfajl+=0

                        continue
                else:
                    nemaanot+=1
                    continue
            elif 'miladinovci' in dir_name:
                break
                # print('miladinovci')
                image = cv2.imread(os.path.join(im_path, im_name))
                rows, col = image.shape[:2]
                cropped_image = image[0:rows, 0:1621]
                rows_cr, col_cr = cropped_image.shape[:2]
                resized = cv2.resize(cropped_image, (512, 341), interpolation=cv2.INTER_AREA)

                objects = []
                HScale = 341 / rows_cr
                WScale = 512 / col_cr

                annot_name = im_name[:-4] + '.txt'
                if os.path.exists(os.path.join(annot_path, annot_name)):
                    if os.stat(os.path.join(annot_path, annot_name)).st_size != 0:
                        # print(annot_path)
                        # print(annot_name)
                        objs = np.loadtxt(os.path.join(annot_path, annot_name), delimiter=',', ndmin=2).astype(np.int)

                        for bbox in objs:
                            # print(bbox)
                            if (bbox[4] == 4):
                                object = [int((bbox[0] - 1) * HScale), int((bbox[1] - 1) * WScale), int((bbox[0] - 1 + bbox[2]) * HScale), int((bbox[1] - 1 + bbox[3]) * WScale), 2]
                            elif (bbox[4] == 3):
                                object = [int((bbox[0] - 1) * HScale), int((bbox[1] - 1) * WScale), int((bbox[0] - 1 + bbox[2]) * HScale), int((bbox[1] - 1 + bbox[3]) * WScale), 1]
                            else:
                                object = [int((bbox[0] - 1) * HScale), int((bbox[1] - 1) * WScale), int((bbox[0] - 1 + bbox[2]) * HScale), int((bbox[1] - 1 + bbox[3]) * WScale), int(bbox[4])]
                            objects.append(object)
                        for obj in objects:
                            cv2.rectangle(resized, (obj[1], obj[0]), (obj[3], obj[2]), color=(0, 255, 0), thickness=1)
                            # print("crtam")
                        # cv2.imshow("slika", resized)
                        # cv2.imshow("slika1", im_res)
                        # cv2.waitKey(0)
                        # print(objects)
                        # &NOTE da se isece kolku sto treba od Martina anotaciite i slikite se ovie
                    else:
                        prazenfajl+=0

                        continue
                else:
                    nemaanot+=1

                    continue
            elif 'DGood' in dir_name:
                break
                # print('dgood')

                image = cv2.imread(os.path.join(im_path, im_name))
                rows, col = image.shape[:2]
                cropped_image = image[0:rows, 149:col]
                rows_cr, col_cr = cropped_image.shape[:2]
                resized = cv2.resize(cropped_image, (512, 341), interpolation=cv2.INTER_AREA)

                objects = []
                HScale = 341 / rows_cr
                WScale = 512 / col_cr

                annot_name = im_name[:-4] + '.txt'
                if os.path.exists(os.path.join(annot_path, annot_name)):
                    if os.stat(os.path.join(annot_path, annot_name)).st_size != 0:
                    # print(annot_path)
                    # print(annot_name)
                        objs = np.loadtxt(os.path.join(annot_path, annot_name), delimiter=',', ndmin=2).astype(np.int)

                        for bbox in objs:
                            # print(bbox)
                            bbox[1]=bbox[1]-149
                            bbox[3]=bbox[3]-149
                            if (bbox[4] == 4):
                                object = [int(bbox[0] * HScale), int(bbox[1] * WScale), int(bbox[2] * HScale), int( bbox[3] * WScale), 2]
                            elif (bbox[4] == 3):
                                object = [int(bbox[0] * HScale), int(bbox[1] * WScale), int(bbox[2] * HScale), int( bbox[3] * WScale), 1]
                            else:
                                object = [int(bbox[0] * HScale), int(bbox[1] * WScale), int(bbox[2] * HScale), int( bbox[3] * WScale), int(bbox[4])]
                            objects.append(object)
                        # for obj in objects:
                        #   cv2.rectangle(resized, (obj[1], obj[0]), (obj[3], obj[2]), color=(0, 255, 0), thickness=1)
                        #     # print("crtam")
                        # cv2.imshow("slika", resized)
                        # # cv2.imshow("slika1", im_res)
                        # cv2.waitKey(0)
                    else:
                        prazenfajl+=0
                        continue #ako e prazen fajlot
                else:
                    nemaanot+=1
                    continue #ako ne postoi fajlot
            elif 'MVI' in dir_name:
                break
                # print('MVI')
                image = cv2.imread(os.path.join(im_path, im_name))
                rows, col = image.shape[:2]
                cropped_image = image[0:rows, 149:col]
                rows_cr, col_cr = cropped_image.shape[:2]
                resized = cv2.resize(cropped_image, (512, 341), interpolation=cv2.INTER_AREA)

                objects = []
                HScale = 341 / rows_cr
                WScale = 512 / col_cr


                annot_name = im_name[:-4] + '.txt'
                if os.path.exists(os.path.join(annot_path, annot_name)):
                    if os.stat(os.path.join(annot_path, annot_name)).st_size != 0:
                    # print(annot_path)
                    # print(annot_name)
                        objs = np.loadtxt(os.path.join(annot_path, annot_name), delimiter=',', ndmin=2).astype(np.int)

                        for bbox in objs:
                            if bbox[4]==2 or bbox[4]==4:
                                flag=1
                                break
                            else:
                                bbox[1] = bbox[1] - 149
                                bbox[3] = bbox[3] - 149
                                object = [int(bbox[0] * HScale), int(bbox[1] * WScale), int( bbox[2] * HScale), int(bbox[3] * WScale), bbox[4]]
                            objects.append(object)
                        if (flag == 1):
                            frli+=1
                            continue
                        # else:
                        #     for obj in objects:
                        #       cv2.rectangle(resized, (obj[1], obj[0]), (obj[3], obj[2]), color=(0, 255, 0), thickness=1)
                        #         # print("crtam")
                        #     cv2.imshow("slika", resized)
                        #     # cv2.imshow("slika1", im_res)
                        #     cv2.waitKey(0)
                    else:
                        prazenfajl+=0

                        continue
                else:
                    nemaanot+=1

                    continue

            elif 'M-30' in dir_name:
                # print('m30')
                resized = cv2.imread(os.path.join(im_path, im_name))

                rows, col = resized.shape[:2]
                if im_size != (col, rows):
                    resized = cv2.resize(resized, im_size, interpolation=cv2.INTER_AREA)
                    # flag1 = 1
                # image = image.reshape(image.shape[0], image.shape[1],3)
                # print(1)
                annot_name = str(int(im_name[5:11].lstrip('0')) - 1) + '.xml'
                if os.path.exists(os.path.join(annot_path, annot_name)):
                    if os.stat(os.path.join(annot_path, annot_name)).st_size != 0:
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
                            elif cl == "van":
                                flag=1
                                break
                                # and bb[3] - bb[2] + 1 > 15:
                                # annot = [int(np.round(bb[2])), int(np.round(bb[0])), int(np.round(bb[3])), int(np.round(bb[1])), 1]
                                # objects.append(annot)
                            elif cl == "truck" and bb[3] - bb[2] + 1 > 15:
                                annot = [int(np.round(bb[2])), int(np.round(bb[0])), int(np.round(bb[3])), int(np.round(bb[1])), 2]
                                objects.append(annot)
                            else:
                                continue
                        if(flag==1):
                            frli+=1
                            continue
                        # for obj in objects:
                        #     print("crtam")
                        #     cv2.rectangle(resized, (obj[1], obj[0]), (obj[3], obj[2]), color=(0, 255, 0), thickness=1)
                        #     # print("crtam")
                        # cv2.imshow("slika", resized)
                        # # cv2.imshow("slika1", im_res)
                        # cv2.waitKey(0)
                    else:
                        continue
                else:
                    continue

            #     cv2.rectangle(image, (annot[1], annot[0]), (annot[3], annot[2]), color=(0, 0, 0), thickness=1)
            #
            # cv2.imshow("slika", image)
            # cv2.waitKey(0)
            elif 'tamara' in dir_name:
                break
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
            # else:
            #     break
            # if len(objects)<0:
            #     continue
            #   &NOTE: otkoga ke se popravat kamionite da se vrati ovoj del

            if len(objects) < 0:
                nemapoz+=1
                # shutil.move(os.path.join(im_path, im_name), os.path.join(dst, im_name))
                continue
                    # print            (object)
            # for obj in objects:
            #     cv2.rectangle(resized, (obj[1], obj[0]), (obj[3], obj[2]), color=(0, 255, 0), thickness=1)
            #     # print("crtam")
            # cv2.imshow("slika", resized)
            # # cv2.imshow("slika1", im_res)
            # cv2.waitKey(0)
            bboxes_train=deepcopy(objects)
            # out_class = []
            num_negs_ratio = 3
            obj_masks_train = helper_anchorless_mc.generate_anchor_level_object_masks(bboxes_train, img_dims, anchor_stride)
            out_class, out_reg=helper_anchorless_mc.get_anchorless_ground_truth_data_parallel_optimized(bboxes_train, obj_masks_train[:, :], img_dims, anchor_stride, iou_low, iou_high, num_negs_ratio, num_classes)
            # print(np.shape(out_reg))
            # reg_norm_coef = np.max(np.abs(out_reg))

            # anchorless_genertor_plot.save_results_anchorless_limits_cls(res, resized, (0,0,0), out_class, out_reg, 8, 0.5, reg_norm_coef)
            if out_class is None and out_reg is None:
                # print(1)
                broj+=1
                # shutil.move(os.path.join(im_path, im_name), os.path.join(dst, im_name))
                continue
            else:

                # reg_norm_coef = np.max(np.abs(out_reg))

                # anchorless_genertor_plot.save_results_anchorless_limits_cls(res, resized, (0,0,0), out_class, out_reg, 8, 0.5, reg_norm_coef)
                out_c_rows, out_c_cols, depth=out_class.shape
                out_r_rows, out_r_cols, depth_r=out_reg.shape
                out_class_dims=[out_c_rows, out_c_cols, depth]
                out_reg_dims=[out_r_rows, out_r_cols, depth_r]
                out_class_flat=np.ndarray.flatten(out_class)

                out_reg_flat=np.ndarray.flatten(out_reg)

                if 'M-30' in dir_name:

                    cv2.imwrite(os.path.join(augmented_dst, im_name), resized)

                    filename=str(im_name[:-4]) +'.txt'
                    fid=open(os.path.join(GroundTruthFiles, filename), 'wb+')
                    pickle.dump(out_class_dims,fid)
                    pickle.dump(out_class_flat, fid)
                    pickle.dump(out_reg_dims,fid)
                    pickle.dump(out_reg_flat,fid)
                    fid.close()


                    #blur

                    im_name_b=int(im_name[5:11])
                    im_name_b_1=im_name_b+16910
                    im_name_new='image' + str(im_name_b_1).zfill(6)
                    im_kon=im_name_new+'.jpg'

                    resulting_image = cv2.filter2D(resized, -1, kernel)
                    cv2.imwrite(os.path.join(augmented_dst, im_kon), resulting_image)

                    filename = str(im_name_new) + '.txt'
                    fid_b = open(os.path.join(GroundTruthFiles, filename), 'wb+')
                    pickle.dump(out_class_dims, fid_b)
                    pickle.dump(out_class_flat, fid_b)
                    pickle.dump(out_reg_dims, fid_b)
                    pickle.dump(out_reg_flat, fid_b)
                    fid_b.close()

                    #regular flip

                    im_name_f=im_name_b_1+16910
                    im_name_new_f = 'image' + str(im_name_f).zfill(6)
                    im_kon=im_name_new_f+'.jpg'

                    regular_flipped = cv2.flip(resized, 1)
                    cv2.imwrite(os.path.join(augmented_dst, im_kon), regular_flipped)

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



                    #blurred flip
                    im_name_f1 = im_name_f + 16910
                    im_name_new_f1 = 'image' + str(im_name_f1).zfill(6)
                    im_kon=im_name_new_f1+'.jpg'

                    blured_flipped = cv2.flip(resulting_image, 1)
                    # name = int(im_name[5:11]) + k
                    # im_name1 = 'image' + str(name).zfill(6) + '.jpg'
                    cv2.imwrite(os.path.join(augmented_dst, im_kon), blured_flipped)
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

    print("Frli:", frli)
    print("PrazenFajl:", prazenfajl)
    print("NemaAnot:", nemaanot)
    print("NemaObjekt:", nemapoz)
    print("NemaPoz:", broj)


m=1
# m=67641
GroundTruthFilesTrain=r'D:\Monika\M-30\GroundTruthPairwiseVal'
# GroundTruthFilesTrain=r'D:\KlasifikacijaVozila\Miladinovci\proba-gt'

if not os.path.exists(GroundTruthFilesTrain):
         os.mkdir(GroundTruthFilesTrain)
#
# srcAnnotationsPathVal=r'E:\Science\Monika4\video4_anotacii'
# GroundTruthFilesVal=r'E:\Science\Monika4\GT_val'
# annot_video2=r'E:\Science\Monika4\video4_anotacii'
# dst=r'E:\Science\Monika4\empty_val'
augmented_dst1=r'D:\Monika\M-30\augmented_validacija'
if not os.path.exists(augmented_dst1):
         os.mkdir(augmented_dst1)

root_im=r'\\192.168.1.153\d\KlasifikacijaVozila\M-30\Sliki\Val'
root_annot=r'\\192.168.1.153\d\KlasifikacijaVozila\M-30\Anotacii'

# root_im=r'D:\KlasifikacijaVozila\Miladinovci\proba-sliki'
# root_annot=r'D:\KlasifikacijaVozila\Miladinovci\proba-ne'

read_data_and_generate_gt(root_im, root_annot,(imgDims['cols'], imgDims['rows']),img_depth,GroundTruthFilesTrain,augmented_dst1, m)


#
#
# m1=67641
# GroundTruthFilesTrain1=r'D:\KlasifikacijaVozila\Miladinovci\IOUPairwise\AugmentedGT'
# # GroundTruthFilesTrain1=r'D:\KlasifikacijaVozila\Miladinovci\proba-gt'
#
# if not os.path.exists(GroundTruthFilesTrain1):
#          os.mkdir(GroundTruthFilesTrain1)
# #
# # srcAnnotationsPathVal=r'E:\Science\Monika4\video4_anotacii'
# # GroundTruthFilesVal=r'E:\Science\Monika4\GT_val'
# # annot_video2=r'E:\Science\Monika4\video4_anotacii'
# # dst=r'E:\Science\Monika4\empty_val'
# augmented_dst11=r'D:\KlasifikacijaVozila\Miladinovci\IOUPairwise\AugmentedPhotos'
# # augmented_dst11=r'D:\KlasifikacijaVozila\Miladinovci\IOUPairwise\augm-proba'
# if not os.path.exists(augmented_dst11):
#          os.mkdir(augmented_dst11)
#
# root_im1=r'D:\KlasifikacijaVozila\Miladinovci\Sliki'
# root_annot1=r'D:\KlasifikacijaVozila\Miladinovci\Anotacii'
#
# # root_im1=r'D:\KlasifikacijaVozila\Miladinovci\proba-sliki'
# # root_annot1=r'D:\KlasifikacijaVozila\Miladinovci\proba-ne'
#
# read_data_and_generate_gt(root_im1, root_annot1,(imgDims['cols'], imgDims['rows']),img_depth,GroundTruthFilesTrain1,augmented_dst11, m1)

# read_data_and_generate_gt(os.path.join(srcImagesPath, 'video4_val'),(imgDims['cols'], imgDims['rows']),img_depth,srcAnnotationsPathVal,annot_path_tamara,annot_video2, GroundTruthFilesVal,m)

