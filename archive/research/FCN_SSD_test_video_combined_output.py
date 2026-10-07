"""
Course: Mashinski vid, FEEIT, Spring 2021
Date: 19.04.2022

Description: apply a fully convolutional SSD architecture for object classification and localization to test data
Python version: 3.6
"""

# python imports
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# import imageio
import numpy as np
import cv2
# import matplotlib.pyplot as plt
# import pickle
# from tensorflow.keras.optimizers import Adam
# import tensorflow.keras as keras
from copy import deepcopy
# custom package imports
from helper import utilities_test_cpu
import pickle
# import helper_model1
# import anchorless_genertor_plot
from tensorflow.keras.layers import *
print("imported")
# NOTE da se menuva flagot normalizeAll vo zavisnost od normalizacijata iskoristena pri dobivanje na modelot

# --- flags ---
flag_save_intermediate_output = False
flag_normalizeAll = True
flag_same_dims = False
flag_predict=True
# --- paths ---
# version = 'vinf_400_cls_reg_finetune_cel_so_nokni_samrak_kamera2_vtor_del'

# version = r'Pedestrians_withTranslations_New_2KLASI_281x469_8_stride_HUMANS_3vs1_0.4_0.55_Softmax_CUSTOM_CATEGORICAL_LOSS_LUGE_HN2_celaslika_negativni_annotirani_koli_Pesak_avtopat_BICUBIC'
version = r'Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor1_All_Negatives_Renamed_diou_focal_notnormreg_anotiraniguzva_yolo_jovan_cbam_preklop'
nacrtaj=True
# src_path = r'D:\Monika\Programs\ffmpeg-n4.4-latest-win64-gpl-4.4\bin\zasijamka_belikoli\NVR_ch1_main_20230619064223_20230619064228.dav'

# NOTE: specify destination paths
# srcImagesPath = r'D:\Monika\VideosTest\Frames_kamera30\kamera30'

#srcImagesPath = r'D:\Monika\AICityChallenge\train\S002\c008\frames'
# srcImagesPath = r'D:\Monika\VideosTest\TEST\kam42\kam42'
# srcImagesPath = r'D:\Monika\VideosTest\Frames_Predmet_na_patot'
# srcImagesPath = r'D:\Monika\VideosTest\kam25_test_pesaci\kam25'
# srcImagesPath = r'D:\Monika\VideosTest\Frames_bajkeri_cam21\miladinovci'
# srcImagesPath = r'D:\Monika\VideosTest\Pesak_na_avtopat'

# srcImagesPath=r'D:\Monika\VideosTest\bajkeri1'
# srcImagesPath = r'D:\Monika\VideosTest\vtor_del'
# srcImagesPath = r'\\HP-1060\ForSharingD\VideoFrames\Samrak'
# srcImagesPath = r'D:\Monika\VideosTest\Frames_Utrinsko'
# srcImagesPath = r'\\192..0 168.1.153\d\VideosTest\Frames_Fleki'
# srcImagesPath = r'\\HP-1060\ForSharingD\video3_frames_black'
# srcAnnotationsPathTest = r'E:\Science\Monika4\GT_test'



# src_model_version = r'Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor1_All_Negatives_Renamed_alpha_1_reg_0.01_DIOU_LOSS_1_novi_fp_1_Focal_CATEGORICAL'  # NOTE: da se pishe
src_model_version = r'Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor1_All_Negatives_Renamed_diou_focal_notnormreg_anotiraniguzva_yolo_jovan_cbam'  # NOTE: da se pishe
# srcModelPath = os.path.join(r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Models',src_model_version)
srcModelPath = os.path.join(r'D:\Monika\Models', src_model_version)

dstResultsPath = r'D:\Monika\Results'
# gtDstPath = r'E:\Science\Monika\GT'
# srcAnnotationsPathTest = r'E:\Science\Monika\anchorless_test'

file_path_reg_coef = os.path.join(r'D:\Monika\Models', src_model_version, 'reg_coef')


# create folders to save data from the current execution


# --- variables ---
#imgDims = {'rows': 1080, 'cols': 1920}
# imgDims = {'rows': 540, 'cols': 960}
# imgDims = {'rows': 281, 'cols': 469}
imgDims = {'rows': 341, 'cols': 512}

num_classes=4
img_depth = 1
img_dims = (imgDims['rows'], imgDims['cols'], img_depth)

# --- load and format data ---
# load full dataset into memory - image data and labels
#x_test, images_names = helper_data.read_data_test_pedestrians(srcImagesPath, img_dims, flag_same_dim=True)
#print(x_test.shape)
#print(f'Number of test samples: {x_test.shape[0]}')

# --- prepare ground truth data in required format ---
# NOTE zavisi od modelot
anchor_stride = 8

out_class_list = []
# for filename in os.listdir(srcAnnotationsPathTest):
#     fid1 = open(os.path.join(srcAnnotationsPathTest, filename), 'rb')
#     out_class_dims = pickle.load(fid1)
#     out_class_back = pickle.load(fid1)
#     fid1.close()
#     out_class = np.reshape(out_class_back, (out_class_dims[0], out_class_dims[1], out_class_dims[2]))
#     out_class_list.append(out_class)
# out_class_list = np.array(out_class_list)

# construct model ---
# lr = 0.0001
import tensorflow as tf
if flag_predict:
    # if flag_same_dims:
    #     model = helper_model1.construct_model_anchorless_detector_skip_v1(input_shape=img_dims)
    #     model.load_weights(os.path.join(srcModelPath, 'new', 'model.h5'))

    # else:
    strategy = tf.distribute.MirroredStrategy()

    # Load the model within the strategy scope
    with strategy.scope():
        model = utilities_test_cpu.load_model_cbam(model_path=os.path.join(srcModelPath, 'model.json'),
                                        weights_path=os.path.join(srcModelPath,'model.h5'))  # build model architecture

# compile model
#     model.compile(loss={
#         'out_class': helper_losses.rpn_loss_cls_new,
#         'out_reg': helper_losses.rpn_loss_reg
#
#     },
#         optimizer=Adam(lr=lr),
#         metrics=['accuracy'])

    # filename = 'reg_norm_coef_position_rows.txt'
    # fid1_1 = open(os.path.join(file_path_reg_coef, filename), 'rb')
    # reg_norm_coef_position_rows = pickle.load(fid1_1)
    #
    # fid1_1.close()
    #
    # filename = 'reg_norm_coef_position_cols.txt'
    # fid1_1 = open(os.path.join(file_path_reg_coef, filename), 'rb')
    # reg_norm_coef_position_cols = pickle.load(fid1_1)
    # fid1_1.close()
    #
    # # size
    # filename = 'reg_norm_coef_size_height.txt'
    # fid1_1 = open(os.path.join(file_path_reg_coef, filename), 'rb')
    # reg_norm_coef_size_height = pickle.load(fid1_1)
    # fid1_1.close()
    #
    # filename = 'reg_norm_coef_size_width.txt'
    # fid1_1 = open(os.path.join(file_path_reg_coef, filename), 'rb')
    # reg_norm_coef_size_width = pickle.load(fid1_1)
    # fid1_1.close()
# model.compile(loss={
#                   'out_class': keras.losses.CategoricalCrossentropy()
# },
#               optimizer=Adam(lr=lr),
#               metrics=['accuracy'])

# x_test_1 = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in x_test]
# x_test_1 = np.array(x_test_1)
# x_test_1 = x_test_1.reshape(x_test_1.shape + (1,))
# x_test_1 = x_test

#
#


# --- apply model to test data ---

# for ind, image in enumerate(x_test_1):
#     image = cv2.resize(image, (img_dims[1], img_dims[0]), interpolation=cv2.INTER_AREA)
#     print(image.shape)
#     image_c = deepcopy(image)
#     image = np.reshape(image, (1, image.shape[0], image.shape[1], 1))
#     im_name = images_names[ind]
#     output_cls = model.predict(image, verbose=1)
#     # print(output_cls[0,:,:,1])
#     print(np.max(output_cls[0, :, :, 1]))
#     r = np.where(output_cls > 0.4)
# print(1)
'''


    # plot histogram of output bounding box sizes
    Y_test_pred_flat = output_cls.flatten()

    # plt.hist(Y_test_pred_flat, density=False, bins=100)  # density=False shows counts, True shows density
    # plt.axvline(0.5, color='k', linestyle='dashed', linewidth=1)
    # plt.ylabel('Count')
    # plt.xlabel('Probability values')
    # plt.show()
'''
prob_thr = 0.85
empty_frames=False
thr_clustering = 0.5
color_small = (0, 255, 0)
color_large = (0, 0, 255)
color_nevozilo=(255,0,0)
colors_list = (color_small, color_large,color_nevozilo)
output_branch = 'classifier'
# if empty_frames:
#
#     reg_norm_coef_position_rows = reg_norm_coef_position_cols = reg_norm_coef_size_height = reg_norm_coef_size_width = None

# videos_path=r'D:\Monika\VideosTest\KAMIONI'
# videos_path=r'D:\Monika\VideosTest\koli_fp_ulici_2\kam40'
# videos_path_src=r'D:\Monika\VideosTest\koli_obratni_fp'
# videos_path_src=r'D:\Monika\VideosTest\Vozila\guzva_test'
videos_path_src=r'D:\Monika\VideosTest\Vozila\guzvatmp\test_guzva'

dir_names=os.listdir(videos_path_src)
# dir_name='kam40'
# print(dir_name)


for dir_name_kam in dir_names:
    if '.zip' in dir_name_kam or not 'kam21' in dir_name_kam:
        continue
    videos_names = os.listdir(os.path.join(videos_path_src,dir_name_kam))

    dir_name = dir_name_kam
    print(dir_name)
    for ind,vid_name in enumerate(videos_names):
        if vid_name=='tmp' or vid_name=='sredeni':
            continue
        if not vid_name.endswith('.dav'):
            continue
        #kade da se zacuvaat ramki, za sekoe video neka ima popseben folder

        # vid='kam38_video'+str(ind)
        if not os.path.exists(os.path.join(dstResultsPath, version,vid_name)):
            os.makedirs(os.path.join(dstResultsPath, version,vid_name), exist_ok=True)
        resultsPath = os.path.join(dstResultsPath, version,vid_name)

        results_path_nms = os.path.join(dstResultsPath, version+ '_postprocessing',vid_name )
        if not os.path.exists(results_path_nms):
            os.makedirs(results_path_nms, exist_ok=True)

        src_path=os.path.join(videos_path_src,dir_name_kam,vid_name)
        vid_cap = cv2.VideoCapture(src_path)  # create video capture object
        vid_cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'HEVC'))
        success, frame = vid_cap.read()  # load first frame, populate success variable

        # extract frames
        frame_cnt = 0  # frame counter
        task = 1
        broi=0


        # duration = vid_cap.get(cv2.CAP_PROP_POS_MSEC)
        # frame_count = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        #
        # print(duration)
        # print(frame_count)
        while success:
            success, frame = vid_cap.read()  # load next frame
            # if frame_cnt>=1000:
            #     break
            print(f'Frame number {frame_cnt}. Success: ', success)
            frame_cnt += 1

            # if frame_cnt%5!=0:
            #     continue
            if frame is None:
                continue
            image = deepcopy(frame)
            # cv2.imshow('orig', image)
            # cv2.waitKey(0)
            orig_image=deepcopy(image)
            rows, col = image.shape[:2]
            #cv2.imshow('slika', image)
            #cv2.waitKey(0)
            # image_save=deepcopy(image)
            # image_save = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if 'kam21' in dir_name or 'glumovo' in dir_name:
                cropped_image = image[0:rows, 0:1621]

                left_crop = 0

            if 'kamera2' in dir_name or 'kam23' in dir_name:
                cropped_image = image[0:rows, 160:1782]
                left_crop = 160

            elif 'kam25' in dir_name or 'kam30' in dir_name:
                print(dir_name)
                cropped_image = image[0:rows, 0:1621]
                left_crop = 0

            elif 'kam32' in dir_name:
                cropped_image = image[0:rows, 299:col]
                left_crop = 299

            elif 'kam28' in dir_name or 'kam33' in dir_name:
                cropped_image = image[0:rows, 99:1720]
                left_crop = 99

            elif 'kam36' in dir_name:
                cropped_image = image[0:rows, 189:1810]
                left_crop = 189

            elif 'kam38' in dir_name:
                cropped_image = image[0:rows, 150:1770]
                left_crop = 150

            elif 'kam40' in dir_name:
                cropped_image = image[0:rows, 255:255+1620]
                left_crop = 299

            elif 'kam42' in dir_name:
                cropped_image = image[0:rows, 299:col]
                left_crop = 299

            elif 'kam44' in dir_name:
                cropped_image = image[0:rows, 299:col]
                left_crop = 299

            elif 'kam46' in dir_name:
                cropped_image = image[0:rows, 100:1721]
                left_crop = 100

            elif 'kam48' in dir_name:
                cropped_image = image[0:rows, 299:col]
                left_crop = 299
            # cv2.imshow('sl_crop', cropped_image)
            # cv2.waitKey(0)
            image_save=deepcopy(cropped_image)

            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

            image = cv2.resize(gray, (img_dims[1], img_dims[0]), interpolation=cv2.INTER_AREA)
            image_save = cv2.resize(image_save, (img_dims[1], img_dims[0]), interpolation=cv2.INTER_AREA)
            #     print(image.shape)
            # image_c = deepcopy(image)


            image = np.reshape(image, (1, image.shape[0], image.shape[1], 1))

            # if task == 1:
                # save frame
            # frame_name = 'image' + '_' + str(frame_cnt).zfill(6) + '.bmp' # create file name

            # cv2.imwrite(os.path.join(resultsPath, frame_name), image)  # save frame as PNG file
            # print(os.path.join(resultsPath, frame_name))
            # im_name = images_names[ind]

            if flag_predict:
                output_combined = model.predict(image, verbose=1)
                output_cls = output_combined[:, :, :, :num_classes]
                output_cls_pos = output_cls[:, :, :, :-1]
                output_reg = output_combined[:, :, :, num_classes:]

            #
            # if task == 0:
            #     # display frame
            #     # cv2.imshow('frame', frame)
            #     # cv2.waitKey(0)
            #     pass
            #


            # read new frame

            # --- plot ground truth network output ---

            # output_branch = 'regressor'
            im_name = 'image' + '_' + str(frame_cnt).zfill(6) + '.bmp'
            results_kamioni1=None
            # save_annot_in_orig=True
            # #list of coordinaes [min_r, min_c, max_r, max_c, 6] of annotation to be saved in txt file for class 6(hard negatives)
            coords_list= [[0,0,340,511,6]]
            #
            #
            normalize=False
            reg_norm_coef_position_rows,reg_norm_coef_position_cols, reg_norm_coef_size_height, reg_norm_coef_size_width=1,1,1,1
            if normalize:
                filename = 'reg_norm_coef_position_rows.txt'
                fid1_1 = open(os.path.join(file_path_reg_coef, filename), 'rb')
                reg_norm_coef_position_rows = pickle.load(fid1_1)

                fid1_1.close()

                filename = 'reg_norm_coef_position_cols.txt'
                fid1_1 = open(os.path.join(file_path_reg_coef, filename), 'rb')
                reg_norm_coef_position_cols = pickle.load(fid1_1)
                fid1_1.close()

                # size
                filename = 'reg_norm_coef_size_height.txt'
                fid1_1 = open(os.path.join(file_path_reg_coef, filename), 'rb')
                reg_norm_coef_size_height = pickle.load(fid1_1)
                fid1_1.close()

                filename = 'reg_norm_coef_size_width.txt'
                fid1_1 = open(os.path.join(file_path_reg_coef, filename), 'rb')
                reg_norm_coef_size_width = pickle.load(fid1_1)
                fid1_1.close()

            if empty_frames:
                # cv2.imshow('slika', image_save)
                # cv2.waitKey(0)
                broi=utilities_test_cpu.save_results_anchorless_limits_save_empty_frames(resultsPath, [image_save], [image], [im_name],
                                                                        anchor_stride, broi,coords_list)

            else:
                utilities_test_cpu.save_results_anchorless_limits(resultsPath,  results_path_nms, results_kamioni1, [image_save], [im_name], output_cls, output_reg,
                                                                        anchor_stride, prob_thr, reg_norm_coef_position_rows,
                                                                        reg_norm_coef_position_cols, reg_norm_coef_size_height, reg_norm_coef_size_width, thr_clustering,
                                                                        colors_list, left_crop, nacrtaj, flag_save_coords=True, flag_choose=False)

            # if broi>=1:
            #     break



        # else:
        #     anchorless_genertor_plot.save_results_anchorless_limits_position_size(resultsPath, results_path_nms, results_kamioni1, x_test,images_names, output_cls, output_reg, anchor_stride, prob_thr,reg_norm_coef_position,reg_norm_coef_size,
        #                                                           thr_clustering, colors_list,flag_save_coords=False)
