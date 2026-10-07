"""
Course: Mashinski vid, FEEIT, Spring 2021
Date: 19.04.2022

Description: apply a fully convolutional SSD architecture for object classification and localization to test data
Python version: 3.6
"""
from keras import backend as K

import gc
K.clear_session()
gc.collect()
# python imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
# from keras.optimizers import Adam
import keras
from copy import deepcopy
# custom package imports
import helper_model, helper_data, helper_losses, helper_model1
import anchorless_genertor_plot
kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
]) / 9



# NOTE da se menuva flagot normalizeAll vo zavisnost od normalizacijata iskoristena pri dobivanje na modelot

# --- flags ---
flag_save_intermediate_output = False
flag_normalizeAll = True
flag_same_dims = True
# --- paths ---
# version = 'vinf_400_cls_reg_finetune_cel_so_nokni_samrak_kamera2_vtor_del'

# version = r'Pedestrians_withTranslations_New_2KLASI_281x469_8_stride_HUMANS_3vs1_0.4_0.55_Softmax_CUSTOM_CATEGORICAL_LOSS_LUGE_HN2_celaslika_negativni_annotirani_koli_Pesak_avtopat_BICUBIC'
version = r'Training_PEDESTRIANS_HalfHD_New5_All_Negatives_Renamed_probafplugeloso1'

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

src_model_version = r'Training_PEDESTRIANS_HalfHD_New5_All_Negatives_Renamed'  # NOTE: da se pishe
# src_model_version = r'Training_PEDESTRIANS_HalfHD_New4_All_Negatives_Renamed_focal_loss_binary_gamma2_sigmoid_FP22_1_128filters_last1conv_biggerval_1layerdoublefilt'  # NOTE: da se pishe
# src_model_version = r'Training_PEDESTRIANS_HalfHD_New1_focal_loss_categorical'  # NOTE: da se pishe
# srcModelPath = os.path.join(r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Models',src_model_version)
srcModelPath = os.path.join(r'D:\Monika\Models', src_model_version)

dstResultsPath = r'D:\Monika\Results'
# gtDstPath = r'E:\Science\Monika\GT'
# srcAnnotationsPathTest = r'E:\Science\Monika\anchorless_test'
file_path_reg_coef = os.path.join(r'D:\Monika\Models', src_model_version, 'reg_coef')

# src_path=r'D:\Monika\Programs\ffmpeg-n4.4-latest-win64-gpl-4.4\bin\TestKuceNaPat'
# src_path=r'D:\Monika\VideosTest\pesaci_fp1'
# src_path=r'D:\Monika\VideosTest\pesaci_fp1'
# src_path=r'M:\Monika\test_pesaci\S001'
# src_path=r'D:\Monika\VideosTest\pesaci_fp_test'
# src_path=r'M:\Luge_Miladinovci'
src_path=r'D:\Monika\VideosTest\luge_fp_loso1'
# src_path=r'D:\Monika\VideosTest\pesaci_fp_za_trening'
# src_path=r'D:\Monika\VideosTest\pesaci_test_ovci'
# dst_path_negatives=r'D:\Monika\Pedestrians_Data_FullHD\Senki_FP5'

dst_path_negatives=r'D:\Monika\Pedestrians_Data_FullHD\Senki_FP29'

predict=True
valid=False

# create folders to save data from the current execution
if predict:
    if not os.path.exists(os.path.join(dstResultsPath, version)):
        os.mkdir(os.path.join(dstResultsPath, version))
    resultsPath = os.path.join(dstResultsPath, version)

    results_path_nms = os.path.join(dstResultsPath, version + '_postprocessing')
    if not os.path.exists(results_path_nms):
        os.mkdir(results_path_nms)



# --- variables ---
#imgDims = {'rows': 1080, 'cols': 1920}
# imgDims = {'rows': 281, 'cols': 469}
# imgDims = {'rows': 281, 'cols': 469}
# imgDims = {'rows': 270, 'cols': 480}
imgDims = {'rows': 540, 'cols': 960}
num_classes = 2
img_depth = 1
img_dims = (imgDims['rows'], imgDims['cols'], img_depth)
flag_same_dims=False
# --- load and format data ---
# load full dataset into memory - image data and labels
#x_test, images_names = helper_data.read_data_test_pedestrians(srcImagesPath, img_dims, flag_same_dim=True)
# print(x_test.shape)
# print(f'Number of test samples: {x_test.shape[0]}')

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
lr = 0.0001
#
# if flag_same_dims:
#     model = helper_model1.construct_model_anchorless_detector_Pedestrians_anchorless(input_shape=img_dims)
#     model.load_weights(os.path.join(srcModelPath, 'new', 'model.h5'))
#
# else:
#     model = helper_model.load_model(model_path=os.path.join(srcModelPath, 'model.json'),
#                                     weights_path=os.path.join(srcModelPath, 'model.h5'))  # build model architecture
#
# # compile model
# model.compile(loss={
#     'out_class': helper_losses.rpn_loss_cls_new
#
# },
#     optimizer=Adam(lr=lr),
#     metrics=['accuracy'])

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
prob_thr = 0.7
thr_clustering = 0.3
color_small = (0, 0, 255)
color_large = (255, 0, 0)
colors_list = (color_small, color_large)
coords_list=[0,0,539,959,6]
output_branch = 'classifier'
reg_norm_coef_position_rows = reg_norm_coef_position_cols = reg_norm_coef_size_height = reg_norm_coef_size_width = None

folderi=os.listdir(src_path)
print(folderi)

# model=helper_model1.load_model(os.path.join(srcModelPath, 'model.json'),os.path.join(srcModelPath, 'model.h5'))
if predict:
    if flag_same_dims:
        if valid:
            model = helper_model1.construct_model_anchorless_detector_Pedestrians_anchorless_padding_valid(input_shape=img_dims)
        else:
            model=helper_model1.construct_model_anchorless_detector_Pedestrians_anchorless(input_shape=img_dims)
        model.load_weights(os.path.join(srcModelPath,'model.h5'))
        # model.load_weights(os.path.join(srcModelPath,'checkpoint-014-0.0000.hdf5'))

    else:
        model = helper_model.load_model(model_path=os.path.join(srcModelPath,'model.json'),
                                    weights_path=os.path.join(srcModelPath, 'model.h5'))
        # build model architectureodel = helper_model.load_model(model_path=os.path.join(srcModelPath,'model.json'),
                                    #weights_path=os.path.join(srcModelPath, 'checkpoint-014-0.0000.hdf5'))  # build model architecture

    # compile model
    # model.compile(loss={
    #                   # 'out_class': helper_losses.rpn_loss_cls_new
    #                   'out_class': helper_losses.categorical_focal_loss([[0.25,0.75]],2)
    #                    },
    #               optimizer=Adam(lr=lr),
    #               metrics=['accuracy'])
    #fp

for folder in folderi:
    if 'tmp' in folder:
        continue
    # if not 'ch22' in folder:
    #     continue
    m=0
    videa_names = os.listdir(os.path.join(src_path,folder))
    print(videa_names)
    if predict:
        if not os.path.exists(os.path.join(resultsPath,folder)):
            os.makedirs(os.path.join(resultsPath,folder), exist_ok=True)

    for video_name in videa_names:
        if  '.exe' in video_name:
            continue
        if not predict:
            if not os.path.exists(os.path.join(dst_path_negatives,'Images', folder)):
                os.makedirs(os.path.join(dst_path_negatives, 'Images', folder), exist_ok=True)

                if not os.path.exists(os.path.join(dst_path_negatives, 'Annotations', folder)):
                    os.makedirs(os.path.join(dst_path_negatives, 'Annotations', folder), exist_ok=True)



        # if video_name[-4:]!='.avi' or video_name[-4:]!='.dav':
        #     continue
        vid_cap = cv2.VideoCapture(os.path.join(src_path,folder,video_name))  # create video capture object
        success, frame = vid_cap.read()  # load first frame, populate success variable

        # frame_count = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # fps = vid_cap.get(cv2.CAP_PROP_FPS)
        # duration = frame_count / fps
        # dur2=vid_cap.get(cv2.CAP_PROP_POS_MSEC)
        # print(f'Duration:{dur2}')
        # extract frames
        frame_cnt = 0  # frame counter
        frame_cnt_zapisani=0
        task=1
        while success:

            success, frame = vid_cap.read()  # load next frame
            print(f'Frame number {frame_cnt}. Success: ', success)
            if not success:
                break
            # frame_cnt_zapisani += 1
            frame_cnt += 1
           # if frame_cnt<1700:
           #     continue
           #  if frame_cnt%3!=0:
           #    continue
            frame_cnt_zapisani+=1

            image=frame
            # image = cv2.blur(image, (3,3))
            if not predict:
                img=cv2.resize(image, (960,540), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(dst_path_negatives,'Images',folder,'image'+ str(m).zfill(6)+'.bmp'), img)
                # coords_list = np.array(coords_list)
                #     ime=str(images_names[im_ind][:-4])
                #     ime=ime+'.txt'
                #     print(ime)
                np.savetxt(os.path.join(dst_path_negatives,'Annotations',folder,'image'+ str(m).zfill(6)+'.txt'), [coords_list], delimiter=',', fmt='%i')
                m+=1
            else:
                # write_jpg=os.path.join(resultsPath, folder, video_name[:-4])
                #
                # if not os.path.exists(write_jpg):
                #     os.makedirs(os.path.join(resultsPath,folder,video_name[:-4]), exist_ok=True)
                # cv2.imwrite(os.path.join(write_jpg, 'im_' + str(frame_cnt) + '.bmp'), image)
                # cv2.imwrite(os.path.join(write_jpg, 'im_'+str(frame_cnt)+'.jpg'), image)
                # im1=cv2.imread(os.path.join(write_jpg, 'im_' + str(frame_cnt) + '.bmp'))
                # im2=cv2.imread(os.path.join(write_jpg, 'im_' + str(frame_cnt) + '.jpg'))
                # res=np.sum(np.abs(im1-im2))
                #print(res)
                #image=cv2.imread(os.path.join(dstResultsPath, 'im_'+str(frame_cnt)+'.jpg'))
                image = cv2.resize(image, (img_dims[1], img_dims[0]), interpolation=cv2.INTER_AREA)
                # image = cv2.filter2D(image, -1, kernel)
            # if frame_cnt==115:
            #     cv2.imwrite(dst, image)
            #     continue
            # else:
            #     continue
            #     print(image.shape)
            # image_c = deepcopy(image)
                image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = np.reshape(image, (1, image.shape[0], image.shape[1], 1))
                # im_name = images_names[ind]
                output_cls = model.predict(image, verbose=1)
                # print(f'Max:{max(output_cls)}')


            #
            # if task == 0:
            #     # display frame
            #     # cv2.imshow('frame', frame)
            #     # cv2.waitKey(0)
            #     pass
            #
            # if task == 1:
            #     # save frame
            #     frame_name = 'image' + '_' + str(frame_cnt).zfill(6) + '.bmp'  # create file name
            #     cv2.imwrite(os.path.join(dst_path, frame_name), frame)  # save frame as PNG file

            # read new frame



        # --- plot ground truth network output ---

        # output_branch = 'regressor'
            output_reg=None
            if predict:
                im_name='image' + '_' + str(frame_cnt).zfill(6) + '.bmp'
                zapisi_slika=os.path.join(resultsPath, folder,video_name[:-4])
                if not os.path.exists(zapisi_slika):
                    os.makedirs(os.path.join(resultsPath,folder,video_name[:-4]), exist_ok=True)

                if flag_normalizeAll:
                    anchorless_genertor_plot.save_results_anchorless_limits_pedestrians(zapisi_slika, results_path_nms, image, im_name, output_cls,output_reg, anchor_stride, prob_thr,
                                                                                        reg_norm_coef_position_rows,
                                                                                        reg_norm_coef_position_cols, reg_norm_coef_size_height, reg_norm_coef_size_width, thr_clustering,
                                                                                        colors_list, flag_save_coords=True)

            # if frame_cnt_zapisani>1500:
            #     break



            # else:
            #     anchorless_genertor_plot.save_results_anchorless_limits_position_size(resultsPath, results_path_nms, results_kamioni1, x_test,images_names, output_cls, output_reg, anchor_stride, prob_thr,reg_norm_coef_position,reg_norm_coef_size,
            #                                                           thr_clustering, colors_list,flag_save_coords=False)
