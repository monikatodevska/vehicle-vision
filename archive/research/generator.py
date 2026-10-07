import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
import os
import random
#from tqdm import tqdm
from copy import deepcopy
from joblib import Parallel, delayed
import multiprocessing
import xml.etree.ElementTree as ET
import keras
# custom imports
import sys
sys.path.insert(0, 'E:\Science\Monika\SingleShotDetector\Helpers' )
import helper_postprocessing

class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, list_IDs, image_path, annot_path_train,  anchor_stride, anchor_dims, iou_low, iou_high, im_size,  img_depth, batch_size, exclude_empty, shuffle,  to_fit=True, n_classes=1):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
        self.image_path = image_path
        self.annot_path=annot_path_train
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.im_size = im_size
        # self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.anchor_stride=anchor_stride
        self.anchor_dims=anchor_dims
        self.iou_low=iou_low
        self.iou_high=iou_high
        self.exclude_empty=exclude_empty
        self.img_depth=img_depth
    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        valid_train=[]
        out_class=[]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        # X = self._generate_X(list_IDs_temp)
        num_cores = multiprocessing.cpu_count()

        x_train_orig, bboxes_train = self.read_data_rpn(list_IDs_temp, self.image_path, (self.im_size[1], self.im_size[0]), self.img_depth, self.annot_path, exclude_empty=True, shuffle=True)
        result = Parallel(n_jobs=num_cores, temp_folder="/tmp", max_nbytes=None, backend="multiprocessing")(delayed(self.get_anchor_data_ssd)(bbox_train, self.anchor_dims, self.im_size, self.anchor_stride, self.iou_low, self.iou_high) for bbox_train in bboxes_train)
        for res_ind, results in enumerate(result):
            # print(results[res_ind])
            if result[res_ind] is None:
                print('k')
                continue
            valid_train.append(res_ind)
            out_class.append(results)
        out_class_train = np.array(out_class)
        x_train = []
        for valid_ind in valid_train:
            x_train.append(x_train_orig[valid_ind])
        x_train = np.array(x_train)
        return x_train,out_class_train

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def read_data_rpn(self, list_id, im_path, im_size, im_depth, annot_path, exclude_empty, shuffle):
        """
        load, resize, and normalize image data
        loads images and annotations from one folder
        :param im_path: global path of folder containing images of a data subset [string]
        :param im_size: output dimensions of the images (cols, rows) [tuple]
        :param im_depth: required depth of the loaded images (value: 1 or 3) [int]
        :param shuffle: whether to shuffle input data order [bool]
        :return: images_list - array of normalized depth maps [ndarray]
        object_annotations_list - annotated bounding boxes min_row, min_col, max_row, max_col [list] """
        #im_path_resized=r'E:\Science\Monika\M-30HD_resized'
        images_list = []       # array of normalized images
        object_annotations_list = []       # array of array of bounding boxes for each image
        WScale=800/1200
        HScale=480/720
        # list images in source folder
        for i, ID in enumerate(list_id):
            # Store sample
            # flag1[i], X[i,] = self._load_grayscale_image(os.path.join(image_path, 'image' + str(ID).zfill(6) + '.jpg'), self.img_dims)
            # images_list1.append(X[i,])
            im_name=str('image' + str(ID).zfill(6) + '.jpg')

            flag1 = 0
            #print(im_name)
            #cv2.waitKey(0)
            # --- load image ---
            if not im_name[-4:] != '.bmp':  # exclude system files
                continue

            if im_depth == 3:
                image = cv2.imread(os.path.join(im_path, im_name))
            else:
                image = cv2.imread(os.path.join(im_path, im_name), 0)
            rows, col=image.shape[:2]
            if im_size != (col,rows):
                image = cv2.resize(image, im_size, interpolation=cv2.INTER_AREA)
                flag1=1
            image = image.reshape(image.shape[0], image.shape[1], im_depth)

            # --- load annotations ---
            annot_name = str(int(im_name[5:11].lstrip('0')) - 1) + '.xml'
            #print(annot_name)

            root = ET.parse(os.path.join(annot_path, annot_name)).getroot()

            objects = []    # list of all objects in the image
            #cv2.imwrite(os.path.join(im_path_resized, im_name), image)
            for object in root.findall('object'):

                cl = object.find('class').text

                bb_xml = object.find('bndbox')
                bb = [np.int(bb_xml.find('xmin').text),     # min_col
                      np.int(bb_xml.find('xmax').text),     # max_col
                      np.int(bb_xml.find('ymin').text),     # min_row
                      np.int(bb_xml.find('ymax').text),     # max_row
                      ]
                if flag1==1:
                  # print(1)
                    annot = [int(np.round(bb[2]*WScale)), int(np.round(bb[0]*HScale)), int(np.round(bb[3]*WScale)), int(np.round(bb[1]*HScale))]
                    #annot_n=[bb[2], bb[0], bb[3], bb[1]]
                    # min_row, min_col, max_row, max_col
                # select positive car samples, height > 25px
                    if cl == 'car' and annot[2] - annot[0] + 1 > 20:
                        # print("zemen")
                        objects.append(annot)
                else:
                    annot=[bb[2], bb[0], bb[3], bb[1]]
                    if cl == 'car' and bb[3] - bb[2] + 1 > 20:
                    # if flag1==1:
                    #     annot[0]=int(np.round(annot[0]*WScale))
                    #     annot[1]=int(np.round(annot[1]*HScale))
                    #     annot[2] =int( np.round(annot[2] * WScale))
                    #     annot[3] =int(np.round(annot[3] * HScale))
                        #cv2.rectangle(imgs[img_ind], (bbox[1], bbox[0]), (bbox[3], bbox[2]), color=(0, 0, 0), thickness=1)

                        # cv2.imshow("slika", imgs[img_ind])
                        # cv2.waitKey(0)

                          objects.append(annot)
                # cv2.rectangle(image, (annot[1], annot[0]), (annot[3], annot[2]), color=(0, 0, 0), thickness=1)
               # cv2.rectangle(image, (annot_n[1], annot_n[0]), (annot_n[3], annot_n[2]), color=(255, 0, 0), thickness=1)
               #  cv2.imshow("slika", image)
               #  cv2.waitKey(0)



            # if flag1==1:
            #     for i, anotacija in enumerate(objects):
            #
            if exclude_empty:
                if len(objects) > 0:
                    images_list.append(image)
                    object_annotations_list.append(objects)

            else:
                images_list.append(image)
                object_annotations_list.append(objects)

        if len(images_list) == 0:
            print("No images were read.")
            exit(100)

        if shuffle:
            data = list(zip(images_list, object_annotations_list))
            random.shuffle(data)
            images_list, object_annotations_list = zip(*data)

        images_list = np.array(images_list).astype(np.uint8)

        return images_list, object_annotations_list


    def get_anchor_data_ssd(self, bboxes, anchor_dims, im_size, anchor_stride, iou_low, iou_high):
        """

        :param bboxes: annotated bounding boxes [min_row, min_col, max_row, max_col] [ndarray]
                       # NOTE: ensure the coordinates are integers
        :param anchor_dims: tuple of anchor dimensions - (height, width) [tuple]
        :param img_dims: (rows, cols, depth) [tuple]
        :param anchor_stride: stride along rows and columns [int]
        :param iou_low: [int]
        :param iou_high: [int]
        :return:
        """

        num_anchors = len(anchor_dims)

        output_dims_class = (np.int(im_size[0] / anchor_stride), np.int(im_size[1] / anchor_stride), num_anchors + 1)
        output_dims_reg = (np.int(im_size[0] / anchor_stride), np.int(im_size[1] / anchor_stride), num_anchors * 4)

        output_class = np.zeros(output_dims_class).astype(np.int)
        output_reg = np.zeros(output_dims_reg).astype(np.int)

        # first position of an anchor center
        start_r = np.int(np.round(anchor_stride / 2))
        start_c = np.int(np.round(anchor_stride / 2))

        for output_row, center_row in enumerate(range(start_r, im_size[0], anchor_stride)):  # iterate through rows of centers
            for output_col, center_col in enumerate(range(start_c, im_size[1], anchor_stride)):  # iterate through columns of centers

                for anchor_ind, anchor_dim in enumerate(anchor_dims):  # iterate through different anchor dimensions

                    half_anchor_dim_h = np.int(np.round(anchor_dim[0] / 2))
                    half_anchor_dim_w = np.int(np.round(anchor_dim[1] / 2))

                    for bbox in bboxes:  # iterate through annotated bounding boxes

                        # --- assign classes: calculate IOU, place 1 or 0 at the required position ---
                        anchor = [max(0, center_row - half_anchor_dim_h),
                                  max(0, center_col - half_anchor_dim_w),
                                  min(center_row - half_anchor_dim_h + anchor_dim[0], im_size[0]),
                                  min(center_col - half_anchor_dim_w + anchor_dim[1], im_size[1])]
                        # min_row, min_col, max_row, max_col

                        iou = helper_postprocessing.calc_iou(bbox, anchor)

                        if iou >= iou_high:

                            # positive sample: set class, calculate deltas
                            output_class[output_row, output_col, anchor_ind] = 1

                            # --- set deltas ---
                            # current location minus correct location
                            # delta_r = bbox[0] - anchor[0]
                            # delta_c = bbox[1] - anchor[1]
                            # delta_h = bbox[2] - bbox[0] - anchor_dim[0]
                            # delta_w = bbox[3] - bbox[1] - anchor_dim[1]
                            #
                            # output_reg[output_row, output_col, anchor_ind * 4 + 0] = delta_r
                            # output_reg[output_row, output_col, anchor_ind * 4 + 1] = delta_c
                            # output_reg[output_row, output_col, anchor_ind * 4 + 2] = delta_h
                            # output_reg[output_row, output_col, anchor_ind * 4 + 3] = delta_w

                        if (iou < iou_high) and (iou > iou_low):
                            # IOU between iou_min and iou_max
                            # class - marked 2, deltas - 0
                            output_class[output_row, output_col, anchor_ind] = 2

        # assign background
        for out_row in range(output_class.shape[0]):  # iterate through rows of output
            for out_col in range(output_class.shape[1]):  # iterate through columns of output

                if sum(output_class[out_row, out_col, :]) == 0:
                    # print(out_row, out_col)
                    output_class[out_row, out_col, num_anchors] = 1

        # replace 2s with 0s
        output_class = np.where(output_class == 2, 0, output_class)

        # remove border pixels
        for ind_a, a_dim in enumerate(anchor_dims):
            border_padding = np.int((anchor_dims[ind_a][0] / anchor_stride) / 2) + 1

            output_class[0:border_padding, :, ind_a] = 0
            output_class[output_class.shape[0] - border_padding:, :, ind_a] = 0
            output_class[:, 0:border_padding, ind_a] = 0
            output_class[:, output_class.shape[1] - border_padding:, ind_a] = 0

            # output_reg[0:border_padding, :, ind_a] = 0
            # output_reg[output_class.shape[0] - border_padding:, :, ind_a] = 0
            # output_reg[:, 0:border_padding, ind_a] = 0
            # output_reg[:, output_class.shape[1] - border_padding:, ind_a] = 0


        # --- select negative samples ---

        # count positives and negatives
        num_positives = np.sum(output_class[:, :, 0:num_anchors])

        # find negatives
        negs = output_class[:, :, num_anchors]
        [r, c] = np.where(negs == 1)


        # select negatives to remove
        ind_to_remove = np.arange(len(r))
        np.random.shuffle(ind_to_remove)

        num_neg = min(len(r), num_positives * 10)   # number of positive to negative samples ratio: 1 to 3
        num_to_remove = len(r) - num_neg
        ind_to_remove = ind_to_remove[:num_to_remove]


        # remove negatives
        for ind in ind_to_remove:
            output_class[r[ind], c[ind], :] = 0

        if num_positives > 0:
            # valid.append(i)
            # i=+1
            # return output_class, output_reg / 100
            return output_class
        else:
            return None

