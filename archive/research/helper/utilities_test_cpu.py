import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from copy import deepcopy
import cv2
import  helper_postprocessing
from tensorflow.keras.models import model_from_json
# from tensorflow import *
# import tensorflow as tf
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, GlobalMaxPooling2D,
                                     Reshape, Add, Multiply, Activation, Lambda,
                                     Concatenate, Conv2D)

import tensorflow as tf
from tensorflow.keras.models import model_from_json
def load_model(model_path, weights_path):
    """
    loads a pre-trained model configuration and calculated weights
    :param model_path: path of the serialized model configuration file (.json) [string]
    :param weights_path: path of the serialized model weights file (.h5) [string]
    :return: model - keras model object
    """

    # --- load model configuration ---
    json_file = open(model_path, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)     # load model architecture

    model.load_weights(weights_path)     # load weights

    return model


def load_model_cbam(model_path, weights_path):
    """
    Loads a pre-trained model configuration and weights.
    """
    with open(model_path, 'r') as json_file:
        model_json = json_file.read()

    # Load model with TensorFlow (tf) in the Lambda layer scope
    model = model_from_json(model_json, custom_objects={'tf': tf})

    model.load_weights(weights_path)
    return model

def save_results_anchorless_limits_save_empty_frames(results_path, images_save, images, images_names,  anchor_stride,broi,coords_list):
    """
    plot bounding boxes of detected objects onto test images and save as images
    :param results_path: path of destination folder for all CNN detections [string]
    :param results_path_nms: path of destination folder of prostprocessed data [string]
    :param images: test images [ndarray]
    :param output_cls: output of the classifier [ndarray]
    :param output_reg: output of the regressor [ndarray]
    :param anchor_stride: stride along rows and columns [int]
    :param prob_thr: probability threshold for object classification (range: 0 to 1) [float]
    :param norm_coef_position: normalization coefficient for position regression data [float]
    :param norm_coef_size: normalization coefficient for image dimensions regression data [float]
    :param thr_clustering: iou threshold for combining detections into one cluster [float]
    :param colors_list: list of colors to plot each class [tuple]
    :return: None
    """
    # print(len(images_names))
    # print(np.shape(images))
    img_dims = (images[0].shape[0], images[0].shape[1])  # height, width
    # num_classes = output_cls.shape[3] - 1
    # binarize classifier output probabilities
    # output_cls[output_cls >= prob_thr] = 1
    # output_cls[output_cls < prob_thr] = 0

    # round regressor output and cast to integer pixel values
    # output_reg[:, :, :, 0] = output_reg[:, :, :, 0] * reg_norm_coef_position_rows  # regressor output
    # output_reg[:, :, :, 1] = output_reg[:, :, :, 1] * reg_norm_coef_position_cols  # regressor output
    # output_reg[:, :, :, 2] = output_reg[:, :, :, 2] * reg_norm_coef_size_height  # regressor output
    # output_reg[:, :, :, 3] = output_reg[:, :, :, 3] * reg_norm_coef_size_width  # regressor output

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = np.int(np.round(anchor_stride / 2))
    start_c = np.int(np.round(anchor_stride / 2))

    for im_ind, image in enumerate(images_save):
        image1 = image.copy()
        # coords_list = []

        image_za_nms = deepcopy(image)
        valid_bboxes = []  # list of bounding boxes with positive height and width

        # res = output_cls[im_ind, :, :, :-2]  # classifier output, probability maps for positive objects only

        # [r, c, d] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)

        # if len(r)!=0:
        #     return broi
        # else:

        cv2.imwrite(os.path.join(results_path, str(images_names[im_ind])), image)
        # coords_list=[40,330,90,370,6]
        np.savetxt(os.path.join(results_path,str(images_names[im_ind])[:-4]+'.txt'), coords_list, delimiter=',', fmt='%i')
        broi+=1

    return broi

def save_results_anchorless_limits(results_path, results_path_nms, results_path_kamioni, images_s, images_names, output_cls, output_reg, anchor_stride, prob_thr,
                                   reg_norm_coef_position_rows,
                                   reg_norm_coef_position_cols, reg_norm_coef_size_height, reg_norm_coef_size_width,
                                   thr_clustering, colors_list,  left_crop, nacrtaj, flag_save_coords,flag_choose):
    """
    plot bounding boxes of detected objects onto test images and save as images
    :param results_path: path of destination folder for all CNN detections [string]
    :param results_path_nms: path of destination folder of prostprocessed data [string]
    :param images: test images [ndarray]
    :param output_cls: output of the classifier [ndarray]
    :param output_reg: output of the regressor [ndarray]
    :param anchor_stride: stride along rows and columns [int]
    :param prob_thr: probability threshold for object classification (range: 0 to 1) [float]
    :param norm_coef_position: normalization coefficient for position regression data [float]
    :param norm_coef_size: normalization coefficient for image dimensions regression data [float]
    :param thr_clustering: iou threshold for combining detections into one cluster [float]
    :param colors_list: list of colors to plot each class [tuple]
    :return: None
    """
    # print(len(images_names))
    # print(np.shape(images))
    img_dims = (images_s[0].shape[0], images_s[0].shape[1])  # height, width
    print(images_s[0].shape[0])
    num_classes = output_cls.shape[3] - 1

    # binarize classifier output probabilities
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0

    # round regressor output and cast to integer pixel values
    output_reg[:, :, :, 0] = output_reg[:, :, :, 0] * reg_norm_coef_position_rows  # regressor output
    output_reg[:, :, :, 1] = output_reg[:, :, :, 1] * reg_norm_coef_position_cols  # regressor output
    output_reg[:, :, :, 2] = output_reg[:, :, :, 2] * reg_norm_coef_size_height  # regressor output
    output_reg[:, :, :, 3] = output_reg[:, :, :, 3] * reg_norm_coef_size_width  # regressor output

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = np.int(np.round(anchor_stride / 2))
    start_c = np.int(np.round(anchor_stride / 2))
    images=[]
    # for im in images_s:
    #     slika=im[:,:,0]
    #     images.append(slika)

    for im_ind, image in enumerate(images_s):
        img_orig=deepcopy(image)
        # rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # image=image[:,:,0]
        #image=image.reshape(image.shape+(1,))
        # image_draw=image
        # image_draw = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        coords_list = []
        #cv2.imshow('sl',image)
        image_za_nms = deepcopy(image)
        valid_bboxes = []  # list of bounding boxes with positive height and width

        res = output_cls[im_ind, :, :, :-1]  # classifier output, probability maps for positive objects only

        [r, c, d] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)

        for pred_ind in range(len(r)):  # iterate over positive predictions
            #ako e klasa nevozilo ne crtaj i ne zacucuvaj koordinati
            if d[pred_ind]==2:
                continue
           #print(d[pred_ind])
            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # 4 - 4 dimensions are fine-tuned: r, c (top left corner), h, w
            delta_r = output_reg[im_ind, r[pred_ind], c[pred_ind], 0]
            delta_c = output_reg[im_ind, r[pred_ind], c[pred_ind], 1]
            h_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 2]
            w_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 3]

            # center_row_reg = int(center_row + delta_r)
            # center_col_reg = int(center_col + delta_c)

            h = h_percent * img_dims[0]
            w = w_percent * img_dims[1]
            print(h)
            print(w)
            # bbox top left point
            min_row = np.int(center_row - np.round(h / 2))
            min_col = np.int(center_col - np.round(w / 2))

            # adjust position and size with regressor predictions
            min_row_adj = np.int(min_row + delta_r)
            min_col_adj = np.int(min_col + delta_c)

            max_row_adj = min_row_adj + h
            max_col_adj = min_col_adj + w

            # plot bounding box onto image
            # clip bounding boxes falling out of the image borders
            min_col_adj = max(0, np.int(min_col_adj))
            min_row_adj = max(0, np.int(min_row_adj))
            max_col_adj = min(np.int(max_col_adj), img_dims[1])
            max_row_adj = min(np.int(max_row_adj), img_dims[0])

            center_row_reg = np.int(min_row_adj + np.round(h / 2))
            center_col_reg = np.int(min_col_adj + np.round(w / 2))
            #print(center_row_reg)
            #print(center_col_reg)

            if (max_row_adj - min_row_adj) > 0 and (max_col_adj - min_col_adj) > 0:
                valid_bboxes.append([min_row_adj, min_col_adj, max_row_adj, max_col_adj, d[pred_ind]])

                cv2.circle(image, (center_col, center_row), 1, color=(255, 0, 0), thickness=2)     # plot object centers
                # cv2.circle(image, (center_col_reg, center_row_reg), 1, color=(255, 0, 0), thickness=1)     # plot object centers
                # image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
                cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=colors_list[d[pred_ind]], thickness=1)

                # coords_list.append([min_row_adj, min_col_adj, max_row_adj - min_row_adj, max_col_adj - min_col_adj, 6])
                coords_list.append([min_row_adj, min_col_adj, max_row_adj, max_col_adj])
            # if d[pred_ind]==1:
            else:
                print("losi koord")
                print(min_row_adj,max_row_adj)
        #cv2.imwrite(os.path.join(results_path, images_names[im_ind]), image) #NOTE: ova koga sakam teski vozila da mi izvadi samo

           #NOTE: ova koga sakam teski vozila da mi izvadi samo
        #cv2.imshow('sl2',image)0
        #cv2.waitKey(0)
        if len(r)!=0:
            cv2.imwrite(os.path.join(results_path, str(images_names[im_ind])),image)
        # cv2.imwrite(os.path.join(results_path, str(images_names[im_ind])[:-4]+'_ORIG.bmp'),img_orig)
        #cv2.imwrite(os.path.join(results_path, images_names[im_ind]), image)

            if flag_save_coords:
                coords_list = np.array(coords_list)
                ime = str(images_names[im_ind][:-4])
                ime = ime + '.txt'
                print(ime)
                np.savetxt(os.path.join(results_path, ime), coords_list, delimiter=',', fmt='%i')


            helper_postprocessing.nms_new(image_za_nms, os.path.join(results_path_nms, str(images_names[im_ind])), valid_bboxes, thr_clustering, colors_list, num_classes, left_crop, nacrtaj,flag_save_coords, flag_choose)
        # if isinstance(result, tuple):
        #     img_postproc, finalwindows=result
        #     if flag_choose:
        #         cv2.imshow('slika', img_postproc)
        #         k = cv2.waitKey(0)
        #         if k==ord('Y'):
        #             print('Good')
        #             cv2.imwrite(os.path.join(results_path_nms, str(images_names[im_ind])), img_orig)
        #             np.savetxt(os.path.join(results_path_nms, str(images_names[im_ind][:-4]) + '.txt'), finalwindows, delimiter=',', fmt='%i')

