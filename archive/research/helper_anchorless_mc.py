import os
import numpy as np
import cv2
import random
import copy
import pickle
#from tqdm import tqdm

import xml.etree.ElementTree as ET

# custom imports
import sys
sys.path.insert(0, 'E:\Science\Monika\SingleShotDetector\Helpers' )
import helper_postprocessing


def read_data_rpn(gt_path, im_path, im_size, im_depth, annot_path, exclude_empty, shuffle):
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
    #im_path_resized=r'E:\Science\Monika\M-30HD_resized'
    images_list = []       # array of normalized images
    object_annotations_list = []       # array of array of bounding boxes for each image
    # WScale=800/1200
    # HScale=480/720
    # list images in source folder

    for im_name1 in (os.listdir(gt_path)):
        im_name = str(im_name1[0:-4]) + '.jpg'
        # flag1 = 0
        # print(im_name)
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
        # WScale = 512 / col
        # print(WScale)
        # HScale = 341 / rows
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

              # print(1)
            if cl=='car' and annot[2] - annot[0] + 1 > 15:
                annot = [int(np.round(bb[2])), int(np.round(bb[0])), int(np.round(bb[3])), int(np.round(bb[1])), 0]
                objects.append(annot)
            elif cl=="van" and annot[2] - annot[0] + 1 > 15:
                annot = [int(np.round(bb[2])), int(np.round(bb[0])), int(np.round(bb[3])), int(np.round(bb[1])), 1]
                objects.append(annot)
            elif cl == "truck" and annot[2] - annot[0] + 1 > 15:
                annot = [int(np.round(bb[2])), int(np.round(bb[0])), int(np.round(bb[3])), int(np.round(bb[1])), 2]
                objects.append(annot)
            cv2.rectangle(image, (annot[1], annot[0]), (annot[3], annot[2]), color=(0, 0, 0), thickness=1)

        cv2.imshow("slika", image)
        cv2.waitKey(0)
            #annot_n=[bb[2], bb[0], bb[3], bb[1]]
            # min_row, min_col, max_row, max_col
        # select positive car samples, height > 25px
        #     if cl == 'car' and annot[2] - annot[0] + 1 > 20:
        #         # print("zemen")
        #         objects.append(annot)
            # else:
            #     annot=[bb[2], bb[0], bb[3], bb[1]]
            #     if cl == 'car' and bb[3] - bb[2] + 1 > 20:
                # if flag1==1:
                #     annot[0]=int(np.round(annot[0]*WScale))
                #     annot[1]=int(np.round(annot[1]*HScale))
                #     annot[2] =int( np.round(annot[2] * WScale))
                #     annot[3] =int(np.round(annot[3] * HScale))
                    #cv2.rectangle(imgs[img_ind], (bbox[1], bbox[0]), (bbox[3], bbox[2]), color=(0, 0, 0), thickness=1)

                    # cv2.imshow("slika", imgs[img_ind])
                    # cv2.waitKey(0)

                      # objects.append(annot)
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

    # if shuffle:
    #     data = list(zip(images_list, object_annotations_list))
    #     random.shuffle(data)
    #     images_list, object_annotations_list = zip(*data)

    images_list = np.array(images_list).astype(np.uint8)

    return images_list, object_annotations_list




def generate_anchor_level_object_masks(bboxes, img_dims, anchor_stride):
    """
    generate masks of bounding boxes in the output-level matrices
    :param bboxes: annotated bounding boxes [min_row, min_col, max_row, max_col] [ndarray]  NOTE: per image !!!!!!
    :param img_dims: (rows, cols, depth) [tuple]
    :param anchor_stride: stride along rows and columns [int]
    :return: object_masks - anchor-level masks of object locations [ndarray]
    """

    object_masks_dims = (np.int(img_dims[0] / anchor_stride), np.int(img_dims[1] / anchor_stride))
    object_masks = np.zeros(object_masks_dims).astype(np.int)

    # iterate over ground truth files for each image

    for bbox in bboxes:  # iterate over annotated bounding boxes row1 col1 row2 col2
        if bbox[-1]==6:
            continue
        # calculate bbox coordinates in output matrix
        # NOTE: floor & ceil namesto round
        bbox_out = [max(0, np.int(np.floor(bbox[0] / anchor_stride))),  # min_row
                    max(0, np.int(np.floor(bbox[1] / anchor_stride))),  # min_col
                    min(object_masks_dims[0] - 1, np.int(np.ceil(bbox[2] / anchor_stride))),  # max_row
                    min(object_masks_dims[1] - 1, np.int(np.ceil(bbox[3] / anchor_stride)))]  # max_col

        # fill object mask
        object_masks[bbox_out[0]:bbox_out[2], bbox_out[1]:bbox_out[3]] = 1

    return object_masks


def get_anchorless_ground_truth_data_parallel_optimized(bboxes, obj_mask, img_dims, anchor_stride, iou_low, iou_high, num_negs_ratio, num_classes, negative_mask,negative_mask_racni):
    """
    generate ground truth output for fully convolutional network for object detection
    multi-output, classifier and regressor branch
    :param bboxes: annotated bounding boxes [min_row, min_col, max_row, max_col] [ndarray]
    :param obj_mask: mask of possible object locations (bounding box centers which lie within an object bounding box) [ndarray]
    :param img_dims: dimensions of input images (rows, cols, depth) [tuple]
    :param anchor_stride: stride along rows and columns [int]
    :param iou_low: samples with lower iou with all objects are declared negative (range: 0 to 1) [float]
    :param iou_high: samples with higher iou with an object are declared negative (range: 0 to 1) [float]
    :param num_negs_ratio: select negative samples num_negs_ratio times more than positive samples [int]
    :return: output_cls_arr - ground truth classes [ndarray]
             output_cls_arr - ground truth regression (normalized to [-1, 1]) [ndarray]
             valid_inds - indices of images containing at least one object [list]
             reg_norm_coef - normalization coefficient for ground truth regression data [float]
    """

    output_dims_class = (np.int(img_dims[0] / anchor_stride), np.int(img_dims[1] / anchor_stride), num_classes+1)
    output_dims_reg = (np.int(img_dims[0] / anchor_stride), np.int(img_dims[1] / anchor_stride), 4)

    output_class = np.zeros(output_dims_class).astype(np.int)
    output_reg = np.zeros(output_dims_reg).astype(np.float64)

    # first position of an anchor center
    start_r = np.int(np.round(anchor_stride / 2))
    start_c = np.int(np.round(anchor_stride / 2))

    [r, c] = np.where(obj_mask == 1)    # possible object locations

    # ako nema pozitivni objekti
    # da se proveri dali ima 6ki
    # ako da, da se popolnat negativnite
    # ako ne, return none

    # print('DEBUG: ')
    # print(len(r))
    # print(np.sum(negative_mask_racni))
    # print(bboxes)


    if len(r) == 0:     # nema pozitivni

        if np.sum(negative_mask_racni) > 0:     # ima 6ki, popolni negativni
            [r_negs, c_negs] = np.where(negative_mask_racni[:, :] == 1)
            for i in range(len(r_negs)):
                if np.sum(output_class[r_negs[i], c_negs[i], :-1]) == 0:
                    output_class[r_negs[i], c_negs[i], -1] = 1

            return output_class, output_reg

        else:
            # nema nisto
            return None, None



    # ctr = 0
    # for output_row, center_row in enumerate(range(start_r, img_dims[0], anchor_stride)):  # iterate through rows of anchor centers
    #     for output_col, center_col in enumerate(range(start_c, img_dims[1], anchor_stride)):  # iterate through columns of anchor centers
    #         ctr += 1
    #
    # print(f'Iterations: {len(r) ** 2} {ctr}')

    for object_num in range(len(r)):

        output_row = r[object_num]
        output_col = c[object_num]

        for bbox in bboxes:  # iterate through annotated bounding

            flag_zanigde = False

            # --- assign classes: calculate IOU, place 1 or 0 at the required position ---
            bbox_h = bbox[2] - bbox[0]
            bbox_w = bbox[3] - bbox[1]
            bbox_class=bbox[4]
            # print(type(bbox_class))
            if bbox_h <= 1 or bbox_w <= 1:
                continue
            if bbox_class == 6:
                continue

            if bbox_h < 30:
                iou_high = 0.6
                iou_low = 0.45

            if bbox[1] < 5 or bbox[2] > img_dims[0] - 5:
                # treba dvojka vo dvete klasi
                flag_zanigde = True

            half_bbox_dim_h = np.int(np.round(bbox_h / 2))
            half_bbox_dim_w = np.int(np.round(bbox_w / 2))

            center_row = output_row * anchor_stride + start_r
            center_col = output_col * anchor_stride + start_c

            anchor = [max(0, center_row - half_bbox_dim_h),
                      max(0, center_col - half_bbox_dim_w),
                      min(center_row + half_bbox_dim_h, img_dims[0]),
                      min(center_col + half_bbox_dim_w, img_dims[1])]
            # min_row, min_col, max_row, max_col

            # iou = helper_postprocessing.calc_iou(bbox, anchor)
            iou_h, iou_w = helper_postprocessing.calc_iou_partwise(bbox, anchor)

            if iou_h >= iou_high and iou_w>=iou_high:   # pozitivnite
            # if iou >= iou_high:   # pozitivnite
                # print(iou_h, iou_w)
                # positive sample: set class, calculate deltas

                if flag_zanigde:
                    output_class[output_row, output_col, bbox_class - 1] = 2
                else:
                    output_class[output_row, output_col, bbox_class-1] = 1

                    # set deltas - current location minus correct location
                    delta_r = bbox[0] - anchor[0]
                    delta_c = bbox[1] - anchor[1]

                    h_percent = bbox_h / img_dims[0]
                    w_percent = bbox_w / img_dims[1]

                    output_reg[output_row, output_col, 0] = delta_r
                    output_reg[output_row, output_col, 1] = delta_c
                    output_reg[output_row, output_col, 2] = h_percent
                    output_reg[output_row, output_col, 3] = w_percent

            if ((iou_h < iou_high) and (iou_h > iou_low)) and ((iou_w < iou_high) and (iou_w > iou_low)):
            # if iou < iou_high and iou > iou_low:
                # IOU between iou_min and iou_max
                #print(output_row, output_col, bbox_class-1)
                output_class[output_row, output_col, bbox_class-1] = 2    # temporarily mark class with 2

                if not flag_zanigde:
                    # set deltas
                    # current location minus correct location
                    delta_r = bbox[0] - anchor[0]
                    delta_c = bbox[1] - anchor[1]

                    h_percent = bbox_h / img_dims[0]
                    w_percent = bbox_w / img_dims[1]

                    output_reg[output_row, output_col, 0] = delta_r
                    output_reg[output_row, output_col, 1] = delta_c
                    output_reg[output_row, output_col, 2] = h_percent
                    output_reg[output_row, output_col, 3] = w_percent

    # mark negative samples
    for out_row in range(output_class.shape[0]):  # iterate through rows of output
        for out_col in range(output_class.shape[1]):  # iterate through columns of output

            if sum(output_class[out_row, out_col, :]) == 0:     # if no anchors at the specified center is marked with 1 (positive) or 2 (in-between)
                output_class[out_row, out_col, num_classes] = 1

    # replace 2s with 0s
    output_class = np.where(output_class == 2, 0, output_class)


    # --- select negative samples ---
    # count positives and negatives
    num_positives = np.sum(output_class[:, :, 0:num_classes])

    # find negatives
    negs = output_class[:, :, num_classes]
    [r, c] = np.where(negs == 1)

    # select negatives to remove
    ind_to_remove = np.arange(len(r))
    np.random.shuffle(ind_to_remove)

    num_neg = min(len(r), num_positives * num_negs_ratio)   # number of positive to negative samples ratio: 1 to 10
    num_to_remove = len(r) - num_neg
    ind_to_remove = ind_to_remove[:num_to_remove]

    # remove negatives that were not selected
    for ind in ind_to_remove:
        output_class[r[ind], c[ind], :] = 0

    for ind in ind_to_remove:
        output_reg[r[ind], c[ind], :] = 0

    if negative_mask is not None:
        [r,c]=np.where(negative_mask[:,:]==1)
        for i in range(len(r)):
            if(np.sum(output_class[r[i],c[i],:-1])==0):
                output_class[r[i],c[i],-1]=1

        # output_class[:,:,-1] = np.where(negative_mask == 1, 1, output_class[:,:,-1])

    if np.sum(negative_mask_racni)>0:
        [r, c] = np.where(negative_mask_racni[:, :] == 1)
        for i in range(len(r)):
            if (np.sum(output_class[r[i], c[i], :-1]) == 0):
                output_class[r[i], c[i], -1] = 1

    # num_negatives = np.sum(output_class[:, :, -1])

    if num_positives > 0:
    # if np.sum(output_class) > 0:

        return output_class, output_reg
    else:
        return None, None


def get_anchorless_ground_truth_data_parallel_optimized_1persample(bboxes, obj_mask, img_dims, anchor_stride, iou_low, iou_high, num_negs_ratio, num_classes, negative_mask,negative_mask_racni):
    """
    generate ground truth output for fully convolutional network for object detection
    multi-output, classifier and regressor branch
    :param bboxes: annotated bounding boxes [min_row, min_col, max_row, max_col] [ndarray]
    :param obj_mask: mask of possible object locations (bounding box centers which lie within an object bounding box) [ndarray]
    :param img_dims: dimensions of input images (rows, cols, depth) [tuple]
    :param anchor_stride: stride along rows and columns [int]
    :param iou_low: samples with lower iou with all objects are declared negative (range: 0 to 1) [float]
    :param iou_high: samples with higher iou with an object are declared negative (range: 0 to 1) [float]
    :param num_negs_ratio: select negative samples num_negs_ratio times more than positive samples [int]
    :return: output_cls_arr - ground truth classes [ndarray]
             output_cls_arr - ground truth regression (normalized to [-1, 1]) [ndarray]
             valid_inds - indices of images containing at least one object [list]
             reg_norm_coef - normalization coefficient for ground truth regression data [float]
    """

    iou_high = 1.1
    iou_low = 0.3

    output_dims_class = (np.int(img_dims[0] / anchor_stride), np.int(img_dims[1] / anchor_stride), num_classes+1)
    output_dims_reg = (np.int(img_dims[0] / anchor_stride), np.int(img_dims[1] / anchor_stride), 4)

    output_class = np.zeros(output_dims_class).astype(np.int)
    output_reg = np.zeros(output_dims_reg).astype(np.float64)

    # first position of an anchor center
    start_r = np.int(np.round(anchor_stride / 2))
    start_c = np.int(np.round(anchor_stride / 2))

    [r, c] = np.where(obj_mask == 1)    # possible object locations

    # ako nema pozitivni objekti
    # da se proveri dali ima 6ki
    # ako da, da se popolnat negativnite
    # ako ne, return none

    # print('DEBUG: ')
    # print(len(r))
    # print(np.sum(negative_mask_racni))
    # print(bboxes)

    if len(r) == 0:     # nema pozitivni

        if np.sum(negative_mask_racni) > 0:     # ima 6ki, popolni negativni
            [r_negs, c_negs] = np.where(negative_mask_racni[:, :] == 1)
            for i in range(len(r_negs)):
                if np.sum(output_class[r_negs[i], c_negs[i], :-1]) == 0:
                    output_class[r_negs[i], c_negs[i], -1] = 1

            return output_class, output_reg

        else:
            # nema nisto
            return None, None

    # ctr = 0
    # for output_row, center_row in enumerate(range(start_r, img_dims[0], anchor_stride)):  # iterate through rows of anchor centers
    #     for output_col, center_col in enumerate(range(start_c, img_dims[1], anchor_stride)):  # iterate through columns of anchor centers
    #         ctr += 1
    #
    # print(f'Iterations: {len(r) ** 2} {ctr}')

    for object_num in range(len(r)):

        output_row = r[object_num]
        output_col = c[object_num]

        # postavi pozitivni primeroci
        for bbox in bboxes:

            # center row and col of bounding box
            center_row = np.int(np.round((bbox[0] + bbox[0] + bbox[2]) / 2))
            center_col = np.int(np.round((bbox[1] + bbox[1] + bbox[3]) / 2))

            # center point in output mask
            pt_out = [np.int(np.round(center_row / anchor_stride)),  # row
                      np.int(np.round(center_col / anchor_stride))]  # col

            # clip out of bounds points
            pt_out[0] = max(0, pt_out[0])
            pt_out[0] = min(output_dims_class[0] - 1, pt_out[0])

            pt_out[1] = max(0, pt_out[1])
            pt_out[1] = min(output_dims_class[1] - 1, pt_out[1])

            # fill ground truth matrix
            bbox_class = bbox[4]
            output_class[pt_out[0], pt_out[1], bbox_class - 1] = 1

        for bbox in bboxes:  # iterate through annotated bounding

            flag_zanigde = False

            # --- assign classes: calculate IOU, place 1 or 0 at the required position ---
            bbox_h = bbox[2] - bbox[0]
            bbox_w = bbox[3] - bbox[1]
            bbox_class=bbox[4]
            # print(type(bbox_class))
            if bbox_h <= 1 or bbox_w <= 1:
                continue
            if bbox_class == 6:
                continue

            # if bbox_h < 30:
            #     iou_high = 0.6
            #     iou_low = 0.45

            if bbox[1] < 5 or bbox[2] > img_dims[0] - 5:
                # treba dvojka vo dvete klasi
                flag_zanigde = True

            half_bbox_dim_h = np.int(np.round(bbox_h / 2))
            half_bbox_dim_w = np.int(np.round(bbox_w / 2))

            center_row = output_row * anchor_stride + start_r
            center_col = output_col * anchor_stride + start_c

            anchor = [max(0, center_row - half_bbox_dim_h),
                      max(0, center_col - half_bbox_dim_w),
                      min(center_row + half_bbox_dim_h, img_dims[0]),
                      min(center_col + half_bbox_dim_w, img_dims[1])]
            # min_row, min_col, max_row, max_col

            iou = helper_postprocessing.calc_iou(bbox, anchor)
            # iou_h, iou_w = helper_postprocessing.calc_iou_partwise(bbox, anchor)

            # if iou_h >= iou_high and iou_w>=iou_high:   # pozitivnite
            if iou >= iou_high:   # pozitivnite
                # print(iou_h, iou_w)
                # positive sample: set class, calculate deltas

                if flag_zanigde:
                    output_class[output_row, output_col, bbox_class - 1] = 2
                else:
                    # output_class[output_row, output_col, bbox_class-1] = 1    # set in previous section, one sample per object

                    # set deltas - current location minus correct location
                    delta_r = bbox[0] - anchor[0]
                    delta_c = bbox[1] - anchor[1]

                    h_percent = bbox_h / img_dims[0]
                    w_percent = bbox_w / img_dims[1]

                    output_reg[output_row, output_col, 0] = delta_r
                    output_reg[output_row, output_col, 1] = delta_c
                    output_reg[output_row, output_col, 2] = h_percent
                    output_reg[output_row, output_col, 3] = w_percent

            # if ((iou_h < iou_high) and (iou_h > iou_low)) and ((iou_w < iou_high) and (iou_w > iou_low)):
            if iou < iou_high and iou > iou_low:
                # IOU between iou_min and iou_max
                output_class[output_row, output_col, bbox_class-1] = 2    # temporarily mark class with 2

                if not flag_zanigde:
                    # set deltas
                    # current location minus correct location
                    delta_r = bbox[0] - anchor[0]
                    delta_c = bbox[1] - anchor[1]

                    h_percent = bbox_h / img_dims[0]
                    w_percent = bbox_w / img_dims[1]

                    output_reg[output_row, output_col, 0] = delta_r
                    output_reg[output_row, output_col, 1] = delta_c
                    output_reg[output_row, output_col, 2] = h_percent
                    output_reg[output_row, output_col, 3] = w_percent

    # mark negative samples
    for out_row in range(output_class.shape[0]):  # iterate through rows of output
        for out_col in range(output_class.shape[1]):  # iterate through columns of output

            if sum(output_class[out_row, out_col, :]) == 0:     # if no anchors at the specified center is marked with 1 (positive) or 2 (in-between)
                output_class[out_row, out_col, num_classes] = 1

    # replace 2s with 0s
    output_class = np.where(output_class == 2, 0, output_class)


    # --- select negative samples ---
    # count positives and negatives
    num_positives = np.sum(output_class[:, :, 0:num_classes])

    # find negatives
    negs = output_class[:, :, num_classes]
    [r, c] = np.where(negs == 1)

    # select negatives to remove
    ind_to_remove = np.arange(len(r))
    np.random.shuffle(ind_to_remove)

    num_neg = min(len(r), num_positives * num_negs_ratio)   # number of positive to negative samples ratio: 1 to 10
    num_to_remove = len(r) - num_neg
    ind_to_remove = ind_to_remove[:num_to_remove]

    # remove negatives that were not selected
    for ind in ind_to_remove:
        output_class[r[ind], c[ind], :] = 0

    for ind in ind_to_remove:
        output_reg[r[ind], c[ind], :] = 0

    if negative_mask is not None:
        [r,c]=np.where(negative_mask[:,:]==1)
        for i in range(len(r)):
            if(np.sum(output_class[r[i],c[i],:-1])==0):
                output_class[r[i],c[i],-1]=1

        # output_class[:,:,-1] = np.where(negative_mask == 1, 1, output_class[:,:,-1])

    if np.sum(negative_mask_racni)>0:
        [r, c] = np.where(negative_mask_racni[:, :] == 1)
        for i in range(len(r)):
            if (np.sum(output_class[r[i], c[i], :-1]) == 0):
                output_class[r[i], c[i], -1] = 1

    # num_negatives = np.sum(output_class[:, :, -1])

    if num_positives > 0:
    # if np.sum(output_class) > 0:

        return output_class, output_reg
    else:
        return None, None


def anchor_level_false_positives_masks(bboxes, img_dims, anchor_stride):
    """
    generate masks of bounding boxes in the output-level matrices
    :param bboxes: annotated bounding boxes [min_row, min_col, max_row, max_col] [ndarray]  NOTE: per image !!!!!!
    :param img_dims: (rows, cols, depth) [tuple]
    :param anchor_stride: stride along rows and columns [int]
    :return: negatives_masks - anchor-level locations of negative samples [ndarray]
    """

    object_masks_dims = (len(bboxes), np.int(img_dims[0] / anchor_stride), np.int(img_dims[1] / anchor_stride))
    object_masks = np.zeros(object_masks_dims).astype(np.int)

    for img_ind, img_bboxes in enumerate(bboxes):  # iterate over ground truth files for each image

        # print(img_bboxes)

        for bbox in img_bboxes:  # iterate over annotated bounding boxes row1 col1 row2 col2

            # --- calculate centers of bounding boxes in output matrix ---
            # print(bbox)

            # center row and col of bounding box
            center_row = np.int(np.round((bbox[0] + bbox[0] + bbox[2]) / 2))
            center_col = np.int(np.round((bbox[1] + bbox[1] + bbox[3]) / 2))

            # center point in output mask
            pt_out = [np.int(np.round(center_row / anchor_stride)),  # row
                      np.int(np.round(center_col / anchor_stride))]  # col

            # clip out of bounds points

            pt_out[0] = max(0, pt_out[0])
            pt_out[0] = min(object_masks_dims[0] - 1, pt_out[0])

            pt_out[1] = max(0, pt_out[1])
            pt_out[1] = min(object_masks_dims[1] - 1, pt_out[1])

            # fill object mask
            object_masks[img_ind, pt_out[0], pt_out[1]] = 1

    return object_masks


def anchor_level_false_positives_masks_mindims2(bboxes, img_dims, anchor_stride):
    """
    generate masks of bounding boxes in the output-level matrices
    :param bboxes: annotated bounding boxes [min_row, min_col, max_row, max_col] [ndarray]  NOTE: per image !!!!!!
    :param img_dims: (rows, cols, depth) [tuple]
    :param anchor_stride: stride along rows and columns [int]
    :return: negatives_masks - anchor-level locations of negative samples [ndarray]
    """

    object_mask_dims = (np.int(img_dims[0] / anchor_stride), np.int(img_dims[1] / anchor_stride))
    object_mask = np.zeros(object_mask_dims).astype(np.int)

    # for img_ind, img_bboxes in enumerate(bboxes):  # iterate over ground truth files for each image

        #img_bboxes = np.array(img_bboxes, copy=False, subok=True, ndmin=2)
        # print(img_bboxes)

    for bbox in bboxes:  # iterate over annotated bounding boxes row1 col1 row2 col2

        # --- calculate centers of bounding boxes in output matrix ---
        # print(bbox)

        # center row and col of bounding box
        center_row = np.int(np.round((bbox[0] + bbox[0] + bbox[2]) / 2))
        center_col = np.int(np.round((bbox[1] + bbox[1] + bbox[3]) / 2))

        # center point in output mask
        pt_out = [np.int(np.round(center_row / anchor_stride)),  # row
                  np.int(np.round(center_col / anchor_stride))]  # col

        # clip out of bounds points

        pt_out[0] = max(0, pt_out[0])
        pt_out[0] = min(object_mask_dims[0] - 1, pt_out[0])

        pt_out[1] = max(0, pt_out[1])
        pt_out[1] = min(object_mask_dims[1] - 1, pt_out[1])

        # fill object mask
        object_mask[pt_out[0], pt_out[1]] = 1

    return object_mask


def generate_anchor_level_hard_negatives_masks(bboxes_img_all, img_dims, anchor_stride, annot_size_percent, negs_ratio):
    """
    NOTE: for a single image
    if no class 6 (negatives) is found, the function returns empty (zero) masks
    generate masks of manually annotated negative samplesin the output-level matrices
    :param bboxes_img: annotated bounding boxes [min_row, min_col, max_row, max_col] [ndarray]  NOTE: per image
    :param img_dims: (rows, cols, depth) [tuple]
    :param anchor_stride: stride along rows and columns [int]
    :param annot_size_percent: percent of the annotation size to reduce [float]
    :param negs_ratio: ratio of negative samples to keep [float]
    :return: negatives_masks - anchor-level locations of negative samples [ndarray]
    """

    # filter annotations with class 6 (negatives) only
    bboxes_img = []
    for bbox in bboxes_img_all:
        if bbox[-1] == 6:
            bboxes_img.append(bbox)

    object_masks_dims = (np.int(img_dims[0] / anchor_stride), np.int(img_dims[1] / anchor_stride))
    object_masks = np.zeros(object_masks_dims).astype(np.int)

    for bbox in bboxes_img:  # iterate over annotated bounding boxes row1 col1 row2 col2

        if bbox[-1] != 6:
            continue

        # --- reduce annotation size ---
        # height = bbox[2] - bbox[0]
        # width = bbox[3] - bbox[1]
        # to_reduce_height_px = np.int(np.floor(height * (1 - annot_size_percent) / 2))
        # to_reduce_width_px = np.int(np.floor(width * (1 - annot_size_percent) / 2))

        # bbox = [bbox[0] + to_reduce_height_px,
        #         bbox[1] + to_reduce_width_px,
        #         bbox[2] - to_reduce_height_px,
        #         bbox[3] - to_reduce_width_px]

        # calculate bbox coordinates in output matrix
        bbox_out = [max(0, np.int(np.floor(bbox[0] / anchor_stride))),  # min_row
                    max(0, np.int(np.floor(bbox[1] / anchor_stride))),  # min_col
                    min(object_masks_dims[0] - 1, np.int(np.ceil(bbox[2] / anchor_stride))),  # max_row
                    min(object_masks_dims[1] - 1, np.int(np.ceil(bbox[3] / anchor_stride)))]  # max_col

        # fill object mask
        object_masks[bbox_out[0]:bbox_out[2], bbox_out[1]:bbox_out[3]] = 1

        # --- select negs_ratio of negatives ---
        # count positives and negatives
        num_negatives = np.sum(object_masks)
        #print(num_negatives)

        # find negatives
        [r, c] = np.where(object_masks == 1)

        # select negatives to remove
        ind_to_remove = np.arange(len(r))
        np.random.shuffle(ind_to_remove)

        num_neg = min(len(r), int(num_negatives * negs_ratio))  # number of positive to negative samples ratio: 1 to 10
        num_to_remove = len(r) - num_neg
        ind_to_remove = ind_to_remove[:num_to_remove]

        # remove negatives that were not selected
        for ind in ind_to_remove:
            object_masks[r[ind], c[ind]] = 0

    return object_masks


def shiftImageHorizontally(image, x):
    height, width=image.shape[:2]
    translation_matrix=np.array([
                                [1,0,x],
                                [0,1,0]
                                ], dtype=np.float32)
    translated_image=cv2.warpAffine(src=image, M=translation_matrix, dsize=(width, height))
    part=image[0:height,0:x]
    part = np.fliplr(part)
    translated_image[0:height,0:x]=part
    # cv2.imshow("shifted", translated_image)
    # cv2.waitKey(0)
    return translated_image
