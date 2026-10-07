import os
import numpy as np
import cv2
import random
#from tqdm import tqdm

import xml.etree.ElementTree as ET
import helper_postprocessing
def get_anchor_data_ssd(bbox, anchor_dims, img_dims, anchor_stride, iou_low, iou_high):
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

    output_dims_class = (np.int(img_dims[0] / anchor_stride), np.int(img_dims[1] / anchor_stride), num_anchors + 1)
    output_dims_reg = (np.int(img_dims[0] / anchor_stride), np.int(img_dims[1] / anchor_stride), num_anchors * 4)

    output_class = np.zeros(output_dims_class).astype(np.int)
    output_reg = np.zeros(output_dims_reg).astype(np.int)

    # first position of an anchor center
    start_r = np.int(np.round(anchor_stride / 2))
    start_c = np.int(np.round(anchor_stride / 2))

    for output_row, center_row in enumerate(range(start_r, img_dims[0], anchor_stride)):  # iterate through rows of centers
        for output_col, center_col in enumerate(range(start_c, img_dims[1], anchor_stride)):  # iterate through columns of centers

            for anchor_ind, anchor_dim in enumerate(anchor_dims):  # iterate through different anchor dimensions

                half_anchor_dim_h = np.int(np.round(anchor_dim[0] / 2))
                half_anchor_dim_w = np.int(np.round(anchor_dim[1] / 2))

                # for bbox in bboxes:  # iterate through annotated bounding boxes

                # --- assign classes: calculate IOU, place 1 or 0 at the required position ---
                anchor = [max(0, center_row - half_anchor_dim_h),
                          max(0, center_col - half_anchor_dim_w),
                          min(center_row - half_anchor_dim_h + anchor_dim[0], img_dims[0]),
                          min(center_col - half_anchor_dim_w + anchor_dim[1], img_dims[1])]
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
