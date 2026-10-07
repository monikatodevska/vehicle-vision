"""
Course: Mashinski vid, FEEIT, Spring 2021
Date: 19.04.2022

Description: apply a fully convolutional SSD architecture for object classification and localization to test data
Python version: 3.6
"""
import os

# from FCN_SSD_test_video import dir_name

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# python imports

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from keras.optimizers import Adam

# custom package imports
import helper_model, helper_data, helper_losses
import anchorless_genertor_plot
# import pywt
import helper_model1

# NOTE da se menuva flagot normalizeAll vo zavisnost od normalizacijata iskoristena pri dobivanje na modelot

# --- flags ---
flag_save_intermediate_output = False
flag_normalizeAll = True

# --- paths ---
# version = 'vinf_400_cls_reg_finetune_cel_so_nokni_samrak_kamera2_vtor_del'
version = r'guzva_test_pola_slika_kam33_detekcii_normalni_3vid_245_good'
# NOTE: specify destination paths
# srcImagesPath = r'D:\Monika\VideosTest\Frames_kam23_FP'
# srcImagesPath = r'D:\Monika\VideosTest\Frames_morning_5fps'
# srcImagesPath = r'D:\Monika\VideosTest\Vozila\test_guzva_fp'
# srcImagesPath = r'D:\Monika\VideosTest\Sijamski'
# srcImagesPath = r'D:\Monika\VideosTest\Vozila\obratni_loso_spojuvanje'
srcImagesPath = r'D:\Monika\Results\guzva_test_pola_slika_kam33_3vid'

src_model_version = r'Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor1_All_Negatives_Renamed_diou_focal_notnormreg_anotiraniguzva'  # NOTE: da se pishe
# srcModelPath = os.path.join(r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Models',src_model_version)

srcModelPath = os.path.join(r'D:\Monika\Models', src_model_version)

dstResultsPath = r'D:\Monika\Results'
file_path_reg_coef = os.path.join(r'D:\Monika\Models', src_model_version, 'reg_coef')

dir_names = os.listdir(srcImagesPath)
# napraveni=os.listdir(os.path.join(dstResultsPath,version))
# parameters

# imgDims = {'rows': 171, 'cols': 256}
# imgDims = {'rows': 341, 'cols': 512}
imgDims = {'rows': 245, 'cols': 512}
# imgDims = {'rows': 1080, 'cols': 1920}
img_depth = 1
anchor_stride = 8
img_dims = (imgDims['rows'], imgDims['cols'], img_depth)
lr = 0.0001
prob_thr = 0.85
thr_clustering = 0.3
color_small = (0, 255, 0)
color_large = (0, 0, 255)
color_nevozilo = (255, 0, 0)
colors_list = (color_small, color_large, color_nevozilo)
flag_same_dims = False
wavelet = False
num_classes=4
for dir_name in dir_names:
    # if not 'Kam33' in dir_name:
    #     continue
    if 'kamion' in dir_name or 'tmp' in dir_name or 'postprocessing' in dir_name:
        continue
    print(dir_name)
    # if dir_name in napraveni:
    #     continue
    # create folders to save data from the current execution
    if not os.path.exists(os.path.join(dstResultsPath, version, dir_name)):
        os.makedirs(os.path.join(dstResultsPath, version, dir_name))
    resultsPath = os.path.join(dstResultsPath, version, dir_name)

    results_path_nms = os.path.join(dstResultsPath, version, dir_name + '_postprocessing')
    if not os.path.exists(results_path_nms):
        os.mkdir(results_path_nms)

    # --- load and format data ---
    # load full dataset into memory - image data and labels
    srcImagesPathNew = os.path.join(srcImagesPath, dir_name)
    x_test, x_test_color,images_names,crop_left_list,crop_width_list,full_hd_list = helper_data.read_data_test(srcImagesPathNew, (imgDims['cols'], imgDims['rows']), img_depth, dir_name, flag_same_dims)



    x_test_new = []

    # za predict
    if wavelet:
        for x in x_test:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            coeffs2 = pywt.dwt2(x, 'db1')
            LL, (LH, HL, HH) = coeffs2
            # LL, (LH, HL, HH) = coeffs
            sliki = np.stack((LL, LH, HL, HH), axis=-1)
            # LL = LL.reshape(LL.shape + (1,))
            # LH = LH.reshape(LH.shape + (1,))
            # HL = HL.reshape(HL.shape + (1,))
            # HH = HH.reshape(HH.shape + (1,))
            # sliki = np.concatenate([LL, LH, HL, HH], axis=-1)

            x_test_new.append(sliki)

    x_test_nn = []
    # for im in x_test:
    #     # im=cv2.resize(im, (256, 171), interpolation=cv2.INTER_AREA)
    #     im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #     im = im.reshape(im.shape + (1,))
    #     x_test_nn.append(im)
    # x_test_nn=x_test_nn.reshape
    # print(f'Number of test samples: {x_test.shape[0]}')

    # --- prepare ground truth data in required format ---
    # construct model ---

    # if flag_same_dims:
    #     model = helper_model1.construct_model_anchorless_detector_skip_v1(input_shape=img_dims)
    #     model.load_weights(os.path.join(srcModelPath, 'new', 'model.h5'))
    #
    # else:
    model = helper_model.load_model(model_path=os.path.join(srcModelPath, 'new_245', 'model.json'),
                                    weights_path=os.path.join(srcModelPath, 'new_245', 'model.h5'))  # build model architecture
    # model = helper_model.load_model(model_path=os.path.join(srcModelPath,'model.json'),
    #                                 weights_path=os.path.join(srcModelPath, 'model.h5'))  # build model architecture

    # compile model

    # model_layer = model.get_layer(name='model_3')
    # weights_biases = model_layer.get_weights()

    # model.compile(loss={
    #                   'out_class': helper_losses.rpn_loss_cls_new,
    #                   'out_reg': helper_losses.rpn_loss_reg
    #                    },
    #               optimizer=Adam(lr=lr),
    #               metrics=['accuracy'])
    if wavelet:
        x_test_new = np.array(x_test_new)
    else:
        # x_test_1 = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in x_test]

        x_test_1 = np.array(x_test)
        x_test_1 = x_test_1.astype(np.float64)
        x_test_1 = x_test_1.reshape(x_test_1.shape + (1,))


    if not flag_normalizeAll:
        filename = 'reg_norm_coef_position_rows.txt'
        fid1_1 = open(os.path.join(file_path_reg_coef, filename), 'rb')
        reg_norm_coef_position_rows = pickle.load(fid1_1)

        fid1_1.close()

        reg_norm_coef_size_width = reg_norm_coef_size_height = reg_norm_coef_position_cols = ''
        # filename = 'reg_norm_coef_position.txt'
        # fid1_1 = open(os.path.join(file_path_reg_coef, filename), 'rb')
        # reg_norm_coef_position = pickle.load(fid1_1)
        # fid1_1.close()
        # # NOTE: treba li close?
        #
        # filename = 'reg_norm_coef_size.txt'
        # fid1_2 = open(os.path.join(file_path_reg_coef, filename), 'rb')
        # reg_norm_coef_size = pickle.load(fid1_2)
        # fid1_2.close()
        # # NOTE: treba li close?

    # else:
    #
    #     filename = 'reg_norm_coef_position_rows.txt'
    #     fid1_1 = open(os.path.join(file_path_reg_coef, filename), 'rb')
    #     reg_norm_coef_position_rows = pickle.load(fid1_1)
    #
    #     fid1_1.close()
    #
    #     filename = 'reg_norm_coef_position_cols.txt'
    #     fid1_1 = open(os.path.join(file_path_reg_coef, filename), 'rb')
    #     reg_norm_coef_position_cols = pickle.load(fid1_1)
    #     fid1_1.close()
    #
    #     # size
    #     filename = 'reg_norm_coef_size_height.txt'
    #     fid1_1 = open(os.path.join(file_path_reg_coef, filename), 'rb')
    #     reg_norm_coef_size_height = pickle.load(fid1_1)
    #     fid1_1.close()
    #
    #     filename = 'reg_norm_coef_size_width.txt'
    #     fid1_1 = open(os.path.join(file_path_reg_coef, filename), 'rb')
    #     reg_norm_coef_size_width = pickle.load(fid1_1)
    #     fid1_1.close()

    # --- apply model to test data ---
    if wavelet:

        output_cls, output_reg = model.predict(x_test_new, verbose=1)
    else:
        output_combined = model.predict(x_test_1, verbose=1)
    # print(output_cls[0,:,:,1])
    # print(1)
    '''
    # NOTE: to review and test
    # remove border pixels
    border_padding = np.int((anchor_dims[-1][0] / anchor_stride) / 2) + 1

    output_cls[0:border_padding, :, :] = 0
    output_cls[output_cls.shape[0] - border_padding:, :, :] = 0
    output_cls[:, 0:border_padding, :] = 0
    output_cls[:, output_cls.shape[1] - border_padding:, :] = 0

    output_reg[0:border_padding, :, :] = 0
    output_reg[output_reg.shape[0] - border_padding:, :, :] = 0
    output_reg[:, 0:border_padding, :] = 0
    output_reg[:, output_reg.shape[1] - border_padding:, :] = 0
    '''


    from keras import Model
    # plot histogram of output bounding box sizes
    output_cls=output_combined[:,:,:,:num_classes]
    output_cls_pos = output_cls[:, :, :, :-1]
    output_reg=output_combined[:,:,:,num_classes:]

    output_layer_name = 'out_class'

    # Access the layer by name
    output_layer = model.get_layer(output_layer_name)

    # Create a new model that outputs the output of the desired layer
    desired_output_model = Model(inputs=model.input, outputs=output_layer.output)

    # Get the output from the desired layer by passing input data
    output_cls_proba = desired_output_model.predict(x_test_1)  # input_data should be your input data

    # print(output)


    # print(output_cls.shape)
    # print(output_cls_pos.shape)
    # print(np.min(output_cls_pos))
    # print(np.max(output_cls_pos))
    # [a, b, c, d] = np.where(output_cls_pos > 0.8)
    # print(len(a))
    # print(a, b, c, d)

    Y_test_pred_flat = output_cls_pos.flatten()

    # plt.hist(Y_test_pred_flat, density=False, bins=100)  # density=False shows counts, True shows density
    #  plt.axvline(0.5, color='k', linestyle='dashed', linewidth=1)
    #  plt.ylabel('Count')
    #  plt.xlabel('Probability values')
    #   plt.show()

    if flag_save_intermediate_output:

        # --- create folders ---
        intermed_out_path = os.path.join(resultsPath, 'intermediate_output')
        if not os.path.exists(intermed_out_path):
            os.mkdir(intermed_out_path)

        # raw probability maps, 64-bit precision
        intermed_cls_out_path = os.path.join(resultsPath, 'intermediate_output', 'probability_maps')
        if not os.path.exists(intermed_cls_out_path):
            os.mkdir(intermed_cls_out_path)

        # raw regressor data, 64-bit precision
        intermed_reg_out_path = os.path.join(resultsPath, 'intermediate_output', 'regression')
        if not os.path.exists(intermed_reg_out_path):
            os.mkdir(intermed_reg_out_path)

        # --- save probability maps ---
        output_cls_pos = output_cls[:, :, :, :-2]  # num_images, h, w, channels

        # print(max(output_cls_pos[0]))
        for im_ind, output_cls_sample in enumerate(output_cls_pos):
            [r, c, d] = np.where(output_cls_sample > prob_thr)
            for i in range(len(r)):

                for channel in range(output_cls_sample.shape[-1]):
                    np.savetxt(os.path.join(intermed_cls_out_path, str(images_names[im_ind]).zfill(6) + '_' + str(channel) + '.txt'), output_cls_sample[:, :, channel],
                               delimiter=',')
                # r,c,d=np.where(output_cls_pos>0.7)
        # --- bounding box adjustments ---
        output_reg_pos = output_reg[:, :, :, :-1]  # num_images, h, w, channels

        for im_ind, output_reg_sample in enumerate(output_reg_pos):
            for channel in range(output_reg_sample.shape[-1]):  # 4 channels: delta_r, delta_c, h_percent, w_percent
                np.savetxt(os.path.join(intermed_reg_out_path, str(images_names[im_ind]).zfill(6) + '_' + str(channel) + '.txt'), output_reg_sample[:, :, channel], delimiter=',')

        # --- normalization coefficients as float ---
        # NOTE zosto e ova????

        # f = open(os.path.join(intermed_out_path, 'norm_coef_position.txt'), 'w')
        # f.write(str(reg_norm_coef_position))
        # f.close()
        #
        # f = open(os.path.join(intermed_out_path, 'norm_coef_size.txt'), 'w')
        # f.write(str(reg_norm_coef_size))
        # f.close()

    # --- plot ground truth network output ---
    results_kamioni = os.path.join(dstResultsPath, version)
    if not os.path.exists(os.path.join(results_kamioni, 'kamioni')):
        os.mkdir(os.path.join(results_kamioni, 'kamioni'))
    results_kamioni1 = os.path.join(results_kamioni, 'kamioni')
    distance = False

    reg_norm_coef_position_rows, reg_norm_coef_position_cols, reg_norm_coef_size_height, reg_norm_coef_size_width = 1, 1, 1, 1

    save_prob_maps=False
    if save_prob_maps:
        bsize, rows, cols, depth = output_cls.shape

        # Initialize the flattened vector
        flattened_vector = np.zeros((1 * 42 * 64 * 4,))

        # Flatten the matrix
        index = 0
        for b in range(1):
            for d in range(4):
                for r in range(42):
                    for c in range(64):
                        flattened_vector[index] = output_cls[b, r, c, d]
                        index += 1

        # Check the shape of the flattened vector
        print("Shape of flattened vector:", flattened_vector.shape)

        # flattened_array = output_reg_proba.flatten()
        # flattened_array_cls = output_cls.flatten()


        # Convert the flattened array into a string with probabilities separated by commas
        # probabilities_text = ','.join(map(str, flattened_array))
        probabilities_text_cls = ','.join(map(str, flattened_vector))

        # Write the probabilities to a text file
        # with open(r'D:\Monika\VideosTest\slika_test\mil_res\prob1.txt', 'w+') as file:
        #     file.write(probabilities_text)

        # with open(r'D:\Monika\VideosTest\slika_test\mil_res\prob_cls1.txt', 'w+') as file:
        #     file.write(probabilities_text_cls)

    if distance:

        anchorless_genertor_plot.save_results_anchorless_limits_distance(resultsPath, results_path_nms, x_test, images_names, output_cls, output_reg, anchor_stride, prob_thr,
                                                                         reg_norm_coef_position_rows,
                                                                         reg_norm_coef_position_cols, reg_norm_coef_size_height, reg_norm_coef_size_width, thr_clustering,
                                                                         colors_list, flag_save_coords=True, flag_after_regressor=True, normAll=False)
    else:
        anchorless_genertor_plot.save_results_anchorless_limits(resultsPath, results_path_nms, results_kamioni1, x_test,x_test_color,crop_left_list,crop_width_list,full_hd_list, images_names, output_cls, output_reg,
                                                                anchor_stride, prob_thr, reg_norm_coef_position_rows,
                                                                reg_norm_coef_position_cols, reg_norm_coef_size_height, reg_norm_coef_size_width, thr_clustering,
                                                                colors_list,left_crop=0, nacrtaj=False,flag_save_coords=False,flag_save_coords_full_hd=False,flag_choose=False)
        # anchorless_genertor_plot.save_results_anchorless_limits_position_size(resultsPath, results_path_nms, results_kamioni1, x_test,images_names, output_cls, output_reg, anchor_stride, prob_thr,reg_norm_coef_position,reg_norm_coef_size,
        #                                                       thr_clustering, colors_list,flag_save_coords=False)
    print(1)