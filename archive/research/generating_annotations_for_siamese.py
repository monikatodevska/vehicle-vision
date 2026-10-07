"""
Course: Mashinski vid, FEEIT, Spring 2021
Date: 19.04.2022

Description: apply a fully convolutional SSD architecture for object classification and localization to test data
Python version: 3.6
"""

# python imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from keras.optimizers import Adam
import keras
from copy import deepcopy
# custom package imports
import helper_model, helper_data, helper_losses, helper_model1
import anchorless_genertor_plot

# NOTE da se menuva flagot normalizeAll vo zavisnost od normalizacijata iskoristena pri dobivanje na modelot

# --- flags ---
flag_save_intermediate_output = False
flag_normalizeAll = True
flag_same_dims = False
flag_predict=True
# --- paths ---
# version = 'vinf_400_cls_reg_finetune_cel_so_nokni_samrak_kamera2_vtor_del'

# version = r'Pedestrians_withTranslations_New_2KLASI_281x469_8_stride_HUMANS_3vs1_0.4_0.55_Softmax_CUSTOM_CATEGORICAL_LOSS_LUGE_HN2_celaslika_negativni_annotirani_koli_Pesak_avtopat_BICUBIC'
version = r'Koli_sijamka_site_kameri_part2'
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

src_model_version = r'Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor1_All_Negatives_Renamed_alpha_1_reg_0.01_DIOU_LOSS_1_novi_fp_1_Focal_CATEGORICAL'  # NOTE: da se pishe
# srcModelPath = os.path.join(r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Models',src_model_version)
srcModelPath = os.path.join(r'D:\Monika\Models', src_model_version)

dstResultsPath = r'D:\Monika\Results'
# gtDstPath = r'E:\Science\Monika\GT'
# srcAnnotationsPathTest = r'E:\Science\Monika\anchorless_test'
file_path_reg_coef = os.path.join(r'D:\Monika\Models', src_model_version, 'reg_coef')


# create folders to save data from the current execution


# --- variables ---

imgDims = {'rows': 341, 'cols': 512}

num_classes=4
img_depth = 1
img_dims = (imgDims['rows'], imgDims['cols'], img_depth)



# --- prepare ground truth data in required format ---
# NOTE zavisi od modelot
anchor_stride = 8

out_class_list = []


# construct model ---
lr = 0.0001
if flag_predict:
    if flag_same_dims:
        model = helper_model1.construct_model_anchorless_detector_skip_v1(input_shape=img_dims)
        model.load_weights(os.path.join(srcModelPath, 'new', 'model.h5'))

    else:
        model = helper_model.load_model(model_path=os.path.join(srcModelPath,'new', 'model.json'),
                                        weights_path=os.path.join(srcModelPath, 'new','model.h5'))  # build model architecture

#
'''
'''
prob_thr = 0.8
empty_frames=False
thr_clustering = 0.3
color_small = (0, 255, 0)
color_large = (0, 0, 255)
color_nevozilo=(255,0,0)
colors_list = (color_small, color_large,color_nevozilo)
output_branch = 'classifier'

videos_path_src=r'D:\Monika\VideosTest\koli_sijamka'

dir_names=os.listdir(videos_path_src)

for dir_name_kam in dir_names:
    if '.zip' in dir_name_kam:
        continue
    videos_names = os.listdir(os.path.join(videos_path_src,dir_name_kam))

    dir_name = dir_name_kam[-5:]
    print(dir_name)
    for ind,vid_name in enumerate(videos_names):
        if vid_name=='tmp' or vid_name=='sredeni':
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



        while success:
            success, frame = vid_cap.read()  # load next frame
            # if frame_cnt>=1000:
            #     break
            print(f'Frame number {frame_cnt}. Success: ', success)
            frame_cnt += 1

            # if frame_cnt%3!=0:
            #     continue
            if frame is None:
                continue
            image = deepcopy(frame)

            orig_image=deepcopy(image)
            rows, col = image.shape[:2]
            #cv2.imshow('slika', image)
            #cv2.waitKey(0)
            # image_save=deepcopy(image)
            # image_save = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if 'kam21' in dir_name:
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
            reg_norm_coef_position_rows,reg_norm_coef_position_cols, reg_norm_coef_size_height, reg_norm_coef_size_width=1,1,1,1



            anchorless_genertor_plot.save_results_anchorless_limits_save_original(resultsPath,  results_path_nms, results_kamioni1, [image_save],[frame], [im_name], output_cls, output_reg,
                                                                        anchor_stride, prob_thr, reg_norm_coef_position_rows,
                                                                        reg_norm_coef_position_cols, reg_norm_coef_size_height, reg_norm_coef_size_width, thr_clustering,
                                                                        colors_list, left_crop,nacrtaj, flag_save_coords=True, flag_choose=False)

            # if broi>=1:
            #     break



        # else:
        #     anchorless_genertor_plot.save_results_anchorless_limits_position_size(resultsPath, results_path_nms, results_kamioni1, x_test,images_names, output_cls, output_reg, anchor_stride, prob_thr,reg_norm_coef_position,reg_norm_coef_size,
        #                                                           thr_clustering, colors_list,flag_save_coords=False)
