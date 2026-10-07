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
import shutil
# import pywt
# custom package imports
import helper_model, helper_data, helper_losses
import anchorless_genertor_plot

#NOTE da se menuva flagot normalizeAll vo zavisnost od normalizacijata iskoristena pri dobivanje na modelot
x_test_all=[]
images_names_all=[]
# --- flags ---
flag_save_intermediate_output = False
flag_after_regressor = True
flag_save_coords=True
wavelet=False
stacked_flag=True
combined_output=True
flag_nms=True
normalize=False
# --- paths ---
# version = 'vinf_400_cls_reg_finetune_cel_so_nokni_samrak_kamera2_vtor_del'
# version = r'v1500_SO_DVA_INCEPTION_SKIP_48_0.6_0.7_normall_NOVO_ORIGINAL_v2_reg'

# NOTE: specify destination paths
srcImagesPath = r'D:\Monika\VideosTest\All_Cameras_annnotated\Images\Renamed'
# srcImagesPath1 = r'D:\Monika\VideosTest\Frames_morning_10fps\miladinovci'
# annotacii=r'D:\Monika\VideosTest\Frames_morning_10fps\anotacii'
# srcImagesPath = r'D:\Monika\VideosTest\vtor_del'
# srcImagesPath = r'\\HP-1060\ForSharingD\VideoFrames\Samrak'
# srcImagesPath = r'D:\Monika\VideosTest\Frames_Utrinsko'
# srcImagesPath = r'\\192..0 168.1.153\d\VideosTest\Frames_Fleki'
# srcImagesPath = r'\\HP-1060\ForSharingD\video3_frames_black'
# srcAnnotationsPathTest = r'E:\Science\Monika4\GT_test'

# src_model_version = r'v1500_SO_DVA_INCEPTION_SKIP_48_0.6_0.7_normall_NOVO_ORIGINAL_v2'   #NOTE: da se pishe
# srcModelPath = os.path.join(r'D:\Monika\Models',src_model_version)

ModelsRootPath=r'D:\Monika\Models\ModelsTestingNew3'
models=os.listdir(ModelsRootPath)
# dstResultsPath = r'D:\Monika\Results\TEST_ALL_CAMERAS2\novmodel_postproc'


results_path_reg = r'D:\Monika\Results\TEST_ALL_CAMERAS2\reg'
results_path_class = r'D:\Monika\Results\TEST_ALL_CAMERAS2\class'
file_path_r = r'D:\Monika\Results\stats'
annot_path = r'D:\Monika\VideosTest\All_Cameras_annnotated\Annotations\Renamed'


dir_names=os.listdir(srcImagesPath)

# if not os.path.exists(dstResultsPath):
#     os.makedirs(dstResultsPath, exist_ok=True)
# create folders to save data from the current execution

# srcAnnotImagesPath=r'D:\Monika\VideosTest\Frames_AllCameras'
# dstImages=r'D:\Monika\VideosTest\All_Cameras_annnotated'
# dir_annots=[x for x in os.listdir(srcAnnotImagesPath) if 'anotacii' in x]
#
# for dir in dir_annots:
#     if not os.path.exists(os.path.join(dstImages, dir[:-9])):
#         os.makedirs(os.path.join(dstImages, dir[:-9]), exist_ok=True)
#         print(dir[:-9])
#     annot_filenames=os.listdir(os.path.join(srcAnnotImagesPath, dir))
#     for annot_filename in annot_filenames:
#         if os.path.exists(os.path.join(srcAnnotImagesPath, dir[:-9],annot_filename[:-4]+'.bmp')):
#
#             shutil.copy(os.path.join(srcAnnotImagesPath,dir[:-9], annot_filename[:-4] + '.bmp'), os.path.join(dstImages, dir[:-9], annot_filename[:-4] + '.bmp'))


# --- variables ---
imgDims = {'rows': 341, 'cols': 512}
num_classes = 4
img_depth = 3
img_dims = (imgDims['rows'], imgDims['cols'], img_depth)

x_test_nn_all=[]

for dir_name in dir_names:
    print(dir_name)
    x_test_new=[]
    x_test_nn=[]

    #move images that have annotations only:
    # annots=os.listdir(os.path.join(srcannotacii)
    # for anot_name in annots:
    #     shutil.copy(os.path.join(srcImagesPath1,anot_name[:-4]+'.bmp'),os.path.join(srcImagesPath,'miladinovci',anot_name[:-4]+'.bmp'))


    # --- load and format data ---
    # load full dataset into memory - image data and labels

    #NOTE privremeno
    ############
    # dir_name='miladinovci'

    ###########
    x_test, images_names = helper_data.read_data_test(os.path.join(srcImagesPath,dir_name), (imgDims['cols'], imgDims['rows']), img_depth,dir_name,flag_same_dims=False)

    for im in x_test:
        # im=cv2.resize(im, (256, 171), interpolation=cv2.INTER_AREA)
        # im=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im=im.reshape(im.shape+(1,))
        x_test_nn.append(im)

    if wavelet:
        for x in x_test:
            if stacked_flag:
                x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

                coeffs_db1 = pywt.dwt2(x, 'db1')
                coeffs_db2 = pywt.dwt2(x, 'db2')
                coeffs_db3 = pywt.dwt2(x, 'db3')
                coeffs_db4 = pywt.dwt2(x, 'db4')

                LL_db1, (LH_db1, HL_db1, HH_db1) = coeffs_db1
                LL_db2, (LH_db2, HL_db2, HH_db2) = coeffs_db2
                LL_db3, (LH_db3, HL_db3, HH_db3) = coeffs_db3
                LL_db4, (LH_db4, HL_db4, HH_db4) = coeffs_db4

                LL_db2_resized=cv2.resize(LL_db2, (LH_db1.shape[1], LH_db1.shape[0]))
                LH_db2_resized = cv2.resize(LH_db2, (LH_db1.shape[1], LH_db1.shape[0]))
                HL_db2_resized = cv2.resize(HL_db2, (HL_db1.shape[1], HL_db1.shape[0]))
                HH_db2_resized = cv2.resize(HH_db2, (HH_db1.shape[1], HH_db1.shape[0]))

                LL_db3_resized=cv2.resize(LL_db3, (LH_db1.shape[1], LH_db1.shape[0]))
                LH_db3_resized = cv2.resize(LH_db3, (LH_db1.shape[1], LH_db1.shape[0]))
                HL_db3_resized = cv2.resize(HL_db3, (HL_db1.shape[1], HL_db1.shape[0]))
                HH_db3_resized = cv2.resize(HH_db3, (HH_db1.shape[1], HH_db1.shape[0]))


                LL_db4_resized=cv2.resize(LL_db4, (LH_db1.shape[1], LH_db1.shape[0]))
                LH_db4_resized = cv2.resize(LH_db4, (LH_db1.shape[1], LH_db1.shape[0]))
                HL_db4_resized = cv2.resize(HL_db4, (HL_db1.shape[1], HL_db1.shape[0]))
                HH_db4_resized = cv2.resize(HH_db4, (HH_db1.shape[1], HH_db1.shape[0]))

                stacked = np.dstack((LL_db1, LH_db1, HL_db1, HH_db1, LL_db2_resized,LH_db2_resized, HL_db2_resized, HH_db2_resized, LL_db3_resized,LH_db3_resized, HL_db3_resized, HH_db3_resized, LL_db4_resized,LH_db4_resized,
                                     HL_db4_resized, HH_db4_resized))
                x_test_new.append(stacked)
            else:
                x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

                coeffs2 = pywt.dwt2(x, 'haar')
                LL, (LH, HL, HH) = coeffs2
                # LL, (LH, HL, HH) = coeffs
                sliki = np.stack((LL, LH, HL, HH), axis=-1)
                # LL = LL.reshape(LL.shape + (1,))
                # LH = LH.reshape(LH.shape + (1,))
                # HL = HL.reshape(HL.shape + (1,))
                # HH = HH.reshape(HH.shape + (1,))
                # sliki = np.concatenate([LL, LH, HL, HH], axis=-1)

                x_test_new.append(sliki)

        x_test_all.extend(x_test_new)
        x_test_nn_all.extend(x_test_nn)
        images_names_all.extend(images_names)

    else:
        x_test_all.extend(x_test)
        images_names_all.extend(images_names)
    # print(f'Number of test samples: {x_test.shape[0]}')


    # --- prepare ground truth data in required format ---

anchor_stride = 8
out_class_list=[]

x_test_nn=[]


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
if wavelet:

    x_test_all = np.array(x_test_all)

else:
    x_test_all = np.array(x_test_all)


    print(x_test_all.shape)
    # x_test_1 = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in x_test_all]
    x_test_1 = [x for x in x_test_all]
    x_test_1 = np.array(x_test_1)
    x_test_1 = x_test_1.reshape(x_test_1.shape + (1,))

index_col=0
#print(x_test_1.shape)
for model_name in models:
    index_col+=1

    # if not 'DISTANCE' in model_name:
    #     continue
    # if 'ORIGINAL' not in model_name:
    #     continue
    distance=False

    if 'tmp' in model_name or 'reg_coef' in model_name:
        continue
    if 'DISTANCE' in model_name:
        continue
    print(model_name)
    flag_normalizeAll = True
    src_model_version = model_name
    version = model_name

    if flag_nms:
        if not os.path.exists(os.path.join(results_path_reg, version+'_postprocessing')):
            os.mkdir(os.path.join(results_path_reg, version+'_postprocessing'))
        resultsPath_reg = os.path.join(results_path_reg, version+'_postprocessing')

        if not os.path.exists(os.path.join(results_path_class, version+'_postprocessing')):
            os.mkdir(os.path.join(results_path_class, version+'_postprocessing'))
        resultsPath_class = os.path.join(results_path_class, version+'_postprocessing')
    else:
        if not os.path.exists(os.path.join(results_path_reg, version)):
            os.mkdir(os.path.join(results_path_reg, version))
        resultsPath_reg = os.path.join(results_path_reg, version)

        if not os.path.exists(os.path.join(results_path_class, version)):
            os.mkdir(os.path.join(results_path_class, version))
        resultsPath_class = os.path.join(results_path_class, version)

    #
    # if not os.path.exists(os.path.join(dstResultsPath, version)):
    #     os.mkdir(os.path.join(dstResultsPath, version))
    # resultsPath = os.path.join(dstResultsPath, version)
    #
    # results_path_nms = os.path.join(dstResultsPath, version + '_postprocessing')
    # if not os.path.exists(results_path_nms):
    #     os.mkdir(results_path_nms)

    srcModelPath = os.path.join(ModelsRootPath, src_model_version)

    # gtDstPath = r'E:\Science\Monika\GT'
    # srcAnnotationsPathTest = r'E:\Science\Monika\anchorless_test'
    file_path_reg_coef = os.path.join(r'D:\Monika\Models\ModelsTestingNew3', src_model_version, 'reg_coef')

    # if model_name == '2_INCEPTIONS_SKIP_v1_finetune' or model_name == '2_INCEPTIONS_SKIP_v1':
    #     flag_normalizeAll = False

    model = helper_model.load_model(model_path=os.path.join(srcModelPath,'model.json'),
                                    weights_path=os.path.join(srcModelPath, 'model.h5'))  # build model architecture

    # compile model
    # model.compile(loss={
    #                   'out_class': helper_losses.rpn_loss_cls_new,
    #                   'out_reg':helper_losses.rpn_loss_reg
    #                    },
    #               optimizer=Adam(lr=lr),
    #               metrics=['accuracy'])


    # if not flag_normalizeAll:
    #
    #     filename = 'reg_norm_coef_position.txt'
    #     fid1_1 = open(os.path.join(file_path_reg_coef, filename), 'rb')
    #     reg_norm_coef_position = pickle.load(fid1_1)
    #     fid1_1.close()
    #     # NOTE: treba li close?
    #
    #     filename = 'reg_norm_coef_size.txt'
    #     fid1_2 = open(os.path.join(file_path_reg_coef, filename), 'rb')
    #     reg_norm_coef_size = pickle.load(fid1_2)
    #     fid1_2.close()
    #     # NOTE: treba li close?

    if distance:

        if not flag_normalizeAll:

            filename = 'reg_norm_coef_position_rows.txt'
            fid1_1 = open(os.path.join(file_path_reg_coef, filename), 'rb')
            reg_norm_coef_position_rows = pickle.load(fid1_1)

            fid1_1.close()

            reg_norm_coef_size_width = reg_norm_coef_size_height = reg_norm_coef_position_cols = ''



        else:

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

    else:
        if normalize:
            if not flag_normalizeAll:
                filename = 'reg_norm_coef_position.txt'
                fid1_1 = open(os.path.join(file_path_reg_coef, filename), 'rb')
                reg_norm_coef_position = pickle.load(fid1_1)
                fid1_1.close()
                # NOTE: treba li close?

                filename = 'reg_norm_coef_size.txt'
                fid1_2 = open(os.path.join(file_path_reg_coef, filename), 'rb')
                reg_norm_coef_size = pickle.load(fid1_2)
                fid1_2.close()

            else:
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
        else:
            reg_norm_coef_position_rows,reg_norm_coef_position_cols, reg_norm_coef_size_height, reg_norm_coef_size_width=1,1,1,1


    # --- apply model to test data ---
    if wavelet:

        output_cls, output_reg = model.predict(x_test_all, verbose=1)
    else:

        if combined_output:
            comb_output = model.predict(x_test_1, verbose=1)
        else:
            output_cls, output_reg = model.predict(x_test_1, verbose=1)

    if combined_output:
        output_cls=comb_output[:,:,:,:num_classes]
        output_reg=comb_output[:,:,:,num_classes:]


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


    # plot histogram of output bounding box sizes
    # Y_test_pred_flat = output_cls.flatten()
    #
    # plt.hist(Y_test_pred_flat, density=False, bins=100)  # density=False shows counts, True shows density
    # plt.axvline(0.5, color='k', linestyle='dashed', linewidth=1)
    # plt.ylabel('Count')
    # plt.xlabel('Probability values')
    # plt.show()


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
        output_cls_pos = output_cls[:, :, :, :-1]   # num_images, h, w, channels

        for im_ind, output_cls_sample in enumerate(output_cls_pos):
            for channel in range(output_cls_sample.shape[-1]):
                np.savetxt(os.path.join(intermed_cls_out_path, str(im_ind).zfill(6) + '_' + str(channel) + '.txt'), output_cls_sample[:, :, channel], delimiter=',')

        # --- bounding box adjustments ---
        output_reg_pos = output_reg[:, :, :, :-1]  # num_images, h, w, channels

        for im_ind, output_reg_sample in enumerate(output_reg_pos):
            for channel in range(output_reg_sample.shape[-1]):    # 4 channels: delta_r, delta_c, h_percent, w_percent
                np.savetxt(os.path.join(intermed_reg_out_path, str(im_ind).zfill(6) + '_' + str(channel) + '.txt'), output_reg_sample[:, :, channel], delimiter=',')

        # --- normalization coefficients as float ---
        #NOTE zosto e ova????

        # f = open(os.path.join(intermed_out_path, 'norm_coef_position.txt'), 'w')
        # f.write(str(reg_norm_coef_position))
        # f.close()
        #
        # f = open(os.path.join(intermed_out_path, 'norm_coef_size.txt'), 'w')
        # f.write(str(reg_norm_coef_size))
        # f.close()


    # --- plot ground truth network output ---
    prob_thr = 0.85
    thr_clustering = 0.3
    color_small = (0, 255, 0)
    color_large = (0, 0, 255)
    colors_list = (color_small, color_large)
    # output_branch = 'regressor'
    output_branch = 'classifier'
    # results_kamioni=os.path.join(dstResultsPath,version)
    # if not os.path.exists(os.path.join(results_kamioni,'kamioni')):
    #     os.mkdir(os.path.join(results_kamioni, 'kamioni'))
    # results_kamioni1 = os.path.join(results_kamioni, 'kamioni')
    max_prob_thr=0.95
    best_f1=0
    best_thr=0
    import metrics
    if distance:


            # anchorless_genertor_plot.save_results_anchorless_limits(resultsPath, results_path_nms, results_kamioni1, x_test,images_names, output_cls, output_reg, anchor_stride, prob_thr, reg_norm_coef_position_rows,
            #                                                    reg_norm_coef_position_cols,reg_norm_coef_size_height,reg_norm_coef_size_width, thr_clustering, colors_list,flag_save_coords=True)


            anchorless_genertor_plot.save_results_anchorless_limits_distance(resultsPath,results_path_nms, x_test_all,images_names_all,output_cls,output_reg,anchor_stride,prob_thr,reg_norm_coef_position_rows,
                                                               reg_norm_coef_position_cols,reg_norm_coef_size_height,reg_norm_coef_size_width,thr_clustering,colors_list,flag_save_coords,flag_after_regressor,flag_normalizeAll)
        # else:
        #     # anchorless_genertor_plot.save_results_anchorless_limits_position_size(resultsPath, results_path_nms, results_kamioni1, x_test,images_names, output_cls, output_reg, anchor_stride, prob_thr,reg_norm_coef_position,reg_norm_coef_size,
        #     #                                                       thr_clustering, colors_list,
        #     #                                                         flag_save_coords=True)
        #     anchorless_genertor_plot.save_results_anchorless_limits_cls1(resultsPath,x_test_all,images_names_all,output_cls,output_reg,anchor_stride,prob_thr,reg_norm_coef_position,reg_norm_coef_size,flag_normalizeAll,flag_save_coords,flag_after_regressor)
        #


    else:
        if flag_normalizeAll:
            if wavelet:

            # anchorless_genertor_plot.save_results_anchorless_limits(resultsPath, results_path_nms, results_kamioni1, x_test,images_names, output_cls, output_reg, anchor_stride, prob_thr, reg_norm_coef_position_rows,
            #                                                    reg_norm_coef_position_cols,reg_norm_coef_size_height,reg_norm_coef_size_width, thr_clustering, colors_list,flag_save_coords=True)
                anchorless_genertor_plot.save_results_anchorless_limits_cls(resultsPath, x_test_nn_all, images_names_all, output_cls, output_reg, anchor_stride, prob_thr,
                                                                                 reg_norm_coef_position_rows,
                                                                                 reg_norm_coef_position_cols, reg_norm_coef_size_height, reg_norm_coef_size_width, flag_normalizeAll,
                                                                                 flag_save_coords, flag_after_regressor)
            else:
                # anchorless_genertor_plot.save_results_anchorless_limits_cls_proba(resultsPath, x_test_1, images_names_all,
                #                                                             output_cls, output_reg, anchor_stride,
                #                                                             prob_thr,
                #                                                             reg_norm_coef_position_rows,
                #                                                             reg_norm_coef_position_cols,
                #                                                             reg_norm_coef_size_height,
                #                                                             reg_norm_coef_size_width,flag_normalizeAll,
                #                                                             flag_save_coords, flag_after_regressor
                #                                                             )
                for prob_thr in np.arange(0.4, max_prob_thr, 0.05):
                    print(prob_thr)
                    anchorless_genertor_plot.save_results_anchorless_limits_cls(resultsPath_reg, x_test_1, images_names_all, output_cls, output_reg, anchor_stride, prob_thr,
                                                                                reg_norm_coef_position_rows,
                                                                                reg_norm_coef_position_cols, reg_norm_coef_size_height, reg_norm_coef_size_width,num_classes,colors_list,thr_clustering,
                                                                                flag_save_coords, True,flag_nms)
                    anchorless_genertor_plot.save_results_anchorless_limits_cls(resultsPath_class, x_test_1,
                                                                                images_names_all, output_cls,
                                                                                output_reg, anchor_stride, prob_thr,
                                                                                reg_norm_coef_position_rows,
                                                                                reg_norm_coef_position_cols,
                                                                                reg_norm_coef_size_height,
                                                                                reg_norm_coef_size_width, num_classes,
                                                                                colors_list, thr_clustering,
                                                                                flag_save_coords, False, flag_nms)
                    f1=metrics.calculate_stats(resultsPath_reg, resultsPath_class, annot_path, file_path_r,9,model_name,best_thr,False)
                    if f1 >= best_f1:
                        best_f1 = f1
                        best_thr = prob_thr

                anchorless_genertor_plot.save_results_anchorless_limits_cls(resultsPath_reg, x_test_1, images_names_all,
                                                                            output_cls, output_reg, anchor_stride,
                                                                            best_thr,
                                                                            reg_norm_coef_position_rows,
                                                                            reg_norm_coef_position_cols,
                                                                            reg_norm_coef_size_height,
                                                                            reg_norm_coef_size_width, num_classes,
                                                                            colors_list, thr_clustering,
                                                                            flag_save_coords, True, flag_nms)
                anchorless_genertor_plot.save_results_anchorless_limits_cls(resultsPath_class, x_test_1,
                                                                            images_names_all, output_cls,
                                                                            output_reg, anchor_stride, best_thr,
                                                                            reg_norm_coef_position_rows,
                                                                            reg_norm_coef_position_cols,
                                                                            reg_norm_coef_size_height,
                                                                            reg_norm_coef_size_width, num_classes,
                                                                            colors_list, thr_clustering,
                                                                            flag_save_coords, False, flag_nms)

                f1 = metrics.calculate_stats(resultsPath_reg, resultsPath_class, annot_path, file_path_r, 9, model_name,best_thr,True)



        else:
            # anchorless_genertor_plot.save_results_anchorless_limits_position_size(resultsPath, results_path_nms, results_kamioni1, x_test,images_names, output_cls, output_reg, anchor_stride, prob_thr,reg_norm_coef_position,reg_norm_coef_size,
            #                                                       thr_clustering, colors_list,
            #                                                         flag_save_coords=True)
            anchorless_genertor_plot.save_results_anchorless_limits_cls1(resultsPath, x_test_all, images_names_all, output_cls, output_reg, anchor_stride, prob_thr,
                                                                         reg_norm_coef_position, reg_norm_coef_size, flag_normalizeAll, flag_save_coords, flag_after_regressor)

        print(best_f1)
        print(best_thr)