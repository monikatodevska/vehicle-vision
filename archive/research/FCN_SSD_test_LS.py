"""
Course: Mashinski vid, FEEIT, Spring 2021
Date: 19.04.2022

Description: apply a fully convolutional SSD architecture for object classification and localization to test data
Python version: 3.6
"""

# python imports
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from keras.optimizers import Adam
import keras
from copy import deepcopy
# custom package imports
import helper_model, helper_data, helper_losses,helper_model1
import anchorless_genertor_plot
import tensorflow as tf
#NOTE da se menuva flagot normalizeAll vo zavisnost od normalizacijata iskoristena pri dobivanje na modelot

# --- flags ---
flag_save_intermediate_output = False
flag_normalizeAll=True
flag_same_dims=False
# --- paths ---
# version = 'vinf_400_cls_reg_finetune_cel_so_nokni_samrak_kamera2_vtor_del'

#version = r'Pedestrians_withTranslations_New_2KLASI_281x469_8_stride_HUMANS_3vs1_0.4_0.55_Softmax_CUSTOM_CATEGORICAL_LOSS_LUGE_HN2_celaslika_negativni_annotirani_koli_Pesak_avtopat_BICUBIC'
version = r'Training_PEDESTRIANS_HalfHD_New5_All_Negatives_Renamed_pr'

# NOTE: specify destination paths
#srcImagesPath = r'D:\Monika\VideosTest\Frames_kamera30\kamera30'

#srcImagesPath = r'D:\Monika\VideosTest\Frames_bajkeri_cam28\miladinovci'
# srcImagesPath = r'D:\Monika\VideosTest\Pesak_na_avtopat\COVEK'
# srcImagesPath = r'D:\Letna_Skola_Load\sliki\IMG_0086'
# srcImagesPath = r'M:\Monika\test_pesaci\S001\c001'
# srcImagesPath = r'D:\Monika\VideosTest\luge_sliki_test_fp\OI\nothing'
srcImagesPath = r'D:\Monika\Pedestrians_Data_FullHD'

#srcImagesPath = r'D:\Monika\VideosTest\TEST\kam42\kam42'
# srcImagesPath = r'D:\Monika\VideosTest\Zivotni_ZaTestiranje'
#srcImagesPath = r'D:\Monika\VideosTest\kam25_test_pesaci\kam25'
#srcImagesPath = r'D:\Monika\VideosTest\Frames_bajkeri_cam21\miladinovci'
#srcImagesPath = r'D:\Monika\VideosTest\Pesak_na_avtopat'

# srcImagesPath=r'D:\Monika\VideosTest\bajkeri1'
# srcImagesPath = r'D:\Monika\VideosTest\vtor_del'
# srcImagesPath = r'\\HP-1060\ForSharingD\VideoFrames\Samrak'
# srcImagesPath = r'D:\Monika\VideosTest\Frames_Utrinsko'
# srcImagesPath = r'\\192..0 168.1.153\d\VideosTest\Frames_Fleki'
# srcImagesPath = r'\\HP-1060\ForSharingD\video3_frames_black'
# srcAnnotationsPathTest = r'E:\Science\Monika4\GT_test'

src_model_version = r'Training_PEDESTRIANS_HalfHD_New5_All_Negatives_Renamed'   #NOTE: da se pishe
# srcModelPath = os.path.join(r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Models',src_model_version)
srcModelPath = os.path.join(r'D:\Monika\Models', src_model_version)

dstResultsPath = r'D:\Monika\Results'
# gtDstPath = r'E:\Science\Monika\GT'
# srcAnnotationsPathTest = r'E:\Science\Monika\anchorless_test'
file_path_reg_coef = os.path.join(r'D:\Monika\Models', src_model_version, 'reg_coef')

# create folders to save data from the current execution
if not os.path.exists(os.path.join(dstResultsPath, version)):
    os.mkdir(os.path.join(dstResultsPath, version))
resultsPath = os.path.join(dstResultsPath, version)

results_path_nms = os.path.join(dstResultsPath, version + '_postprocessing')
if not os.path.exists(results_path_nms):
    os.mkdir(results_path_nms)

# --- variables ---
# imgDims = {'rows': 1080, 'cols': 1920}
imgDims = {'rows': 540, 'cols': 960}
# imgDims = {'rows': 375, 'cols': 625}
# imgDims = {'rows': 281, 'cols': 469}
num_classes = 2
img_depth = 1
img_dims = (imgDims['rows'], imgDims['cols'], img_depth)

flag_nvr=False
# --- load and format data ---
# load full dataset into memory - image data and labels


#
#
# x_test,images_paths= helper_data.read_data_test_pedestrians_fp(srcImagesPath, img_dims)
# print(x_test.shape)
# print(f'Number of test samples: {x_test.shape[0]}')
#

# anchor_stride = 8

out_class_list=[]

# construct model ---
lr = 0.0001

if flag_same_dims:
    model=helper_model1.construct_model_anchorless_detector_Pedestrians_anchorless(input_shape=img_dims)
    model.load_weights(os.path.join(srcModelPath,'model.h5'))
    for layer in model.layers:
        print(layer.name, layer.trainable)
        print('k')
else:
    model = helper_model.load_model(model_path=os.path.join(srcModelPath,'model.json'),
                                weights_path=os.path.join(srcModelPath, 'model.h5'))  # build model architecture


# x_test_1=x_test

c=0
im_root=srcImagesPath
dataset_names = os.listdir(im_root)
images = []
images_paths = []
prob_thr = 0.7

thr_clustering = 0.3
multiclass = False

if multiclass:
    color_velosiped = (0, 0, 255)

    color_zivotno = (0, 255, 0)
    # color_tovar = (0, 255, 0)
    colors_list = [color_velosiped, color_zivotno]
else:
    color_pesak = (0, 0, 255)
    colors_list = [color_pesak]

# output_branch = 'regressor'
output_branch = 'classifier'
# output_reg=None
# reg_norm_coef_position_rows=reg_norm_coef_position_cols=reg_norm_coef_size_height=reg_norm_coef_size_width=None
results_kamioni = os.path.join(dstResultsPath, version)
if not os.path.exists(os.path.join(results_kamioni, 'kamioni')):
    os.mkdir(os.path.join(results_kamioni, 'kamioni'))
results_kamioni1 = os.path.join(results_kamioni, 'kamioni')

reg_norm_coef_position_rows = None
reg_norm_coef_position_cols = None
reg_norm_coef_size_height = None
reg_norm_coef_size_width = None
output_reg = None
for dataset in dataset_names:

    if not dataset=="Senki_FP29":
        continue

    print(dataset)

    if dataset[-3:] == '.gz' or dataset[-4:] == '.rar' or dataset[-4:] == '.zip':
        continue

    dir_images_folders = os.listdir(os.path.join(im_root, dataset, 'Images'))  # folderite vo sliki od datasetot
    print(dir_images_folders)
    # if 'Kitti' in dataset or 'CVC' in dataset:
    #     continue

    for dir in dir_images_folders:

        if dir[-3:] == '.gz' or dir[-4:] == '.rar':
            continue

        im_path = os.path.join(im_root, dataset, 'Images', dir)
        images_filenames = os.listdir(im_path)
        for im_name in images_filenames:
            images_paths=[]
            image = cv2.imread(os.path.join(im_path, im_name), 0)

            image = cv2.resize(image, (img_dims[1], img_dims[0]), interpolation=cv2.INTER_AREA)

# for ind, image in enumerate(x_test_1):

    # image = cv2.resize(image, (img_dims[1], img_dims[0]), interpolation=cv2.INTER_AREA)
    # print(image.shape)
            image_c=deepcopy(image)
            image=np.reshape(image,(1,image.shape[0],image.shape[1],1))
            images_paths.append(os.path.join(im_path, im_name))
            output_cls = model.predict(image, verbose=1)
            print(output_cls.shape)



            # plot histogram of output bounding box sizes
            # Y_test_pred_flat = output_cls.flatten()

            # plt.hist(Y_test_pred_flat, density=False, bins=100)  # density=False shows counts, True shows density
            # plt.axvline(0.5, color='k', linestyle='dashed', linewidth=1)
            # plt.ylabel('Count')
            # plt.xlabel('Probability values')
            # plt.show()


            # if flag_save_intermediate_output:
            #
            #     # --- create folders ---
            #     intermed_out_path = os.path.join(resultsPath, 'intermediate_output')
            #     if not os.path.exists(intermed_out_path):
            #         os.mkdir(intermed_out_path)
            #
            #     # raw probability maps, 64-bit precision
            #     intermed_cls_out_path = os.path.join(resultsPath, 'intermediate_output', 'probability_maps')
            #     if not os.path.exists(intermed_cls_out_path):
            #         os.mkdir(intermed_cls_out_path)
            #
            #     # raw regressor data, 64-bit precision
            #     intermed_reg_out_path = os.path.join(resultsPath, 'intermediate_output', 'regression')
            #     if not os.path.exists(intermed_reg_out_path):
            #         os.mkdir(intermed_reg_out_path)
            #
            #     # --- save probability maps ---
            #     output_cls_pos = output_cls[:, :, :, :-1]   # num_images, h, w, channels
            #
            #     for im_ind, output_cls_sample in enumerate(output_cls_pos):
            #         for channel in range(output_cls_sample.shape[-1]):
            #             np.savetxt(os.path.join(intermed_cls_out_path, str(im_ind).zfill(6) + '_' + str(channel) + '.txt'), output_cls_sample[:, :, channel], delimiter=',')
            #
            #     # --- bounding box adjustments ---
            #     output_reg_pos = output_reg[:, :, :, :-1]  # num_images, h, w, channels
            #
            #     for im_ind, output_reg_sample in enumerate(output_reg_pos):
            #         for channel in range(output_reg_sample.shape[-1]):    # 4 channels: delta_r, delta_c, h_percent, w_percent
            #             np.savetxt(os.path.join(intermed_reg_out_path, str(im_ind).zfill(6) + '_' + str(channel) + '.txt'), output_reg_sample[:, :, channel], delimiter=',')
            #
            #     # --- normalization coefficients as float ---
            #     #NOTE zosto e ova????
            #
            #     # f = open(os.path.join(intermed_out_path, 'norm_coef_position.txt'), 'w')
            #     # f.write(str(reg_norm_coef_position))
            #     # f.close()
            #     #
            #     # f = open(os.path.join(intermed_out_path, 'norm_coef_size.txt'), 'w')
            #     # f.write(str(reg_norm_coef_size))
            #     # f.close()


            # --- plot ground truth network output ---


            if flag_normalizeAll:

                anchorless_genertor_plot.save_results_anchorless_limits_pedestrians_original_matrices(images_paths, image, output_cls,prob_thr)
            # else:
            #     anchorless_genertor_plot.save_results_anchorless_limits_position_size(resultsPath, results_path_nms, results_kamioni1, x_test,images_names, output_cls, output_reg, anchor_stride, prob_thr,reg_norm_coef_position,reg_norm_coef_size,
            #                                                           thr_clustering, colors_list,flag_save_coords=False)
