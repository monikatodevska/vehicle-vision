"""
Course: Mashinski vid, FEEIT, Spring 2021
Date: 19.04.2022

Description: apply a fully convolutional SSD architecture for object classification and localization to test data
Python version: 3.6
"""

# python imports
import os
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
# flag_predict=True
# --- paths ---
# version = 'vinf_400_cls_reg_finetune_cel_so_nokni_samrak_kamera2_vtor_del'

# version = r'Pedestrians_withTranslations_New_2KLASI_281x469_8_stride_HUMANS_3vs1_0.4_0.55_Softmax_CUSTOM_CATEGORICAL_LOSS_LUGE_HN2_celaslika_negativni_annotirani_koli_Pesak_avtopat_BICUBIC'
version = 'guzva_test_pola_slika_kam33_3vid'
# dir_name='kam25'
src_path = r'D:\Monika\VideosTest\Vozila\guzva_test_pola_slika'

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

# src_model_version = r'Trening_With_Translations-Up-Down-All_Cams_Nevozilo_Renamed_NOVO'  # NOTE: da se pishe
# srcModelPath = os.path.join(r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Models',src_model_version)
# srcModelPath = os.path.join(r'D:\Monika\Models', src_model_version)

dstResultsPath = r'D:\Monika\Results'
# gtDstPath = r'E:\Science\Monika\GT'
# srcAnnotationsPathTest = r'E:\Science\Monika\anchorless_test'
# file_path_reg_coef = os.path.join(r'D:\Monika\Models', src_model_version, 'reg_coef')


# create folders to save data from the current execution
if not os.path.exists(os.path.join(dstResultsPath, version)):
    os.mkdir(os.path.join(dstResultsPath, version))
resultsPath = os.path.join(dstResultsPath, version)

results_path_nms = os.path.join(dstResultsPath, version + '_postprocessing')
if not os.path.exists(results_path_nms):
    os.mkdir(results_path_nms)

imgDims = {'rows': 341, 'cols': 512}

img_depth = 1
img_dims = (imgDims['rows'], imgDims['cols'], img_depth)


# extract frames
frame_cnt = 0  # frame counter
task = 1
broi=0


# duration = vid_cap.get(cv2.CAP_PROP_POS_MSEC)
# frame_count = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)

# print(duration)
# print(frame_count)



folderi=os.listdir(src_path)
print(folderi)
resized=False
m=0
for dir_name in folderi:
    if 'tmp' in dir_name:
        continue

    videa_names = [x for x in os.listdir(os.path.join(src_path,dir_name)) if os.path.isfile(os.path.join(src_path,dir_name,x))]
    print(videa_names)
    if not os.path.exists(os.path.join(resultsPath,dir_name)):
        os.makedirs(os.path.join(resultsPath,dir_name), exist_ok=True)
    for video_name in videa_names:

        vid_cap = cv2.VideoCapture(os.path.join(src_path, dir_name, video_name))  # create video capture object
        success, frame = vid_cap.read()  # load first frame, populate success variable
        # load first frame, populate success variable
        frame_cnt = 0  # frame counter
        frame_cnt_zapisani = 0
        while success:
            success, frame = vid_cap.read()  # load next frame
            print(f'Frame number {frame_cnt}. Success: ', success)
            if success==False:
                break
            frame_cnt += 1
            # if frame_cnt%3!=0:
            #     continue
            image = frame
            if image is None:
                continue
            rows, col = image.shape[:2]

            if resized:
                if 'kam21' in dir_name:
                    cropped_image = image[0:rows, 0:1621]
                    left_crop = 0

                if 'kamera2' in dir_name or 'kam23' in dir_name or 'ch3' in dir_name:
                    cropped_image = image[0:rows, 160:1782]
                    left_crop = 160
                elif 'kam25' in dir_name or 'ch5' in dir_name:
                    cropped_image = image[0:rows, 0:1621]
                    left_crop = 0
                elif 'kam28' in dir_name or 'ch8' in dir_name:
                    cropped_image = image[0:rows, 149:1770]
                    left_crop = 149
                elif 'kam30' in dir_name or 'ch10' in dir_name:
                    cropped_image = image[0:rows, 0:1621]
                    left_crop = 0

                elif 'kam32' in dir_name or 'ch12' in dir_name:
                    cropped_image = image[0:rows, 299:col]
                    left_crop = 299

                elif 'kam33' in dir_name or 'ch13' in dir_name:
                    cropped_image = image[0:rows, 99:1720]
                    left_crop = 99

                elif 'kam36' in dir_name or 'ch16' in dir_name:
                    cropped_image = image[0:rows, 189:1810]
                    left_crop = 189

                elif 'kam38' in dir_name or 'ch18' in dir_name:
                    cropped_image = image[0:rows, 150:1770]
                    left_crop = 150

                elif 'kam40' in dir_name or 'ch20' in dir_name:
                    cropped_image = image[0:rows, 299:col]
                    left_crop = 299

                elif 'kam42' in dir_name or 'ch22' in dir_name:
                    cropped_image = image[0:rows, 299:col]
                    left_crop = 299

                elif 'kam44' in dir_name or 'ch24' in dir_name:
                    cropped_image = image[0:rows, 299:col]
                    left_crop = 299

                elif 'kam46' in dir_name or 'ch26' in dir_name:
                    cropped_image = image[0:rows, 100:1721]
                    left_crop = 100

                elif 'kam48' in dir_name or 'ch28' in dir_name:
                    cropped_image = image[0:rows, 299:col]
                    left_crop = 299

                image = cv2.resize(cropped_image, (img_dims[1], img_dims[0]), interpolation=cv2.INTER_AREA)
                #     print(image.shape)
                # image_c = deepcopy(image)
                image_save=deepcopy(image)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            im_name = 'image' + '_' + str(m).zfill(6) + '.bmp'

            cv2.imwrite(os.path.join(resultsPath,dir_name, im_name), image)  # save frame as PNG file
            m+=1
            # negative_bbox=[0,0,img_dims[0],img_dims[1],6]
            # np.savetxt(os.path.join(resultsPath,dir_name, im_name[:-4]+'.txt'), [negative_bbox], delimiter=',', fmt='%i')

            # print(os.path.join(resultsPath, frame_name))
            # im_name = images_names[ind]




