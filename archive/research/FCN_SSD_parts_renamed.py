"""
Course: Mashinski vid, FEEIT, Spring 2021
Date: 09.08.2021

Description: design, train and evaluate a fully convolutional SSD architecture for object classification and localization
Python version: 3.6
"""

# python imports
import os
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
# from tensorflow.keras.optimizers import Adam
from keras.optimizers import Adam
from joblib import Parallel, delayed
import multiprocessing
import helper_anchorless
import helper_model1
import pickle
import tensorflow as tf
import anchorless_genertor_plot
import h5py
# custom package imports
# from SingleShotDetector.Helpers import helper_model, helper_data, helper_stats, losses
import sys
# sys.path.insert(0, 'E:\Science\Monika\SingleShotDetector\Helpers' )
# import cv2

import helper_model, helper_stats, helper_losses
import helper_data
import cv2

from keras.utils.multi_gpu_utils import multi_gpu_model
from keras import backend as K
import gc
# print(K.tensorflow_backend._get_available_gpus())   # check available number of gpus


if __name__ == '__main__':

    version = 'Trening_With_Translations-Up-Down-All_Cams_Wavelet_NAJNOVI_Renamed_DB_STACKED_NEW'
    # version = 'Proba_vejv'
    # version = 'vinf_proba_ne'

    # --- paths ---
    srcImagesPath = r'D:\Monika\Trening_With_Translations-Up-Down-All_Cams_Wavelet_NAJNOVI_Renamed\Images'  # ova e eden folder
    GT_train = r'D:\Monika\Trening_With_Translations-Up-Down-All_Cams_Wavelet_NAJNOVI_Renamed\GT'

    srcImagesVal = r'D:\Monika\Trening_With_Translations-Up-Down-All_Cams_Wavelet_NAJNOVI_Renamed\Validacija\Images'
    GT_val = r'D:\Monika\Trening_With_Translations-Up-Down-All_Cams_Wavelet_NAJNOVI_Renamed\Validacija\GT'

    file_path_reg_coef = r'D:\Monika\Trening_With_Translations-Up-Down-All_Cams_Wavelet_NAJNOVI_Renamed\reg_coef' #NOTE da se pise


    # srcImagesPath = r'D:\Monika\Proba_Data\Images'  # ova e eden folder
    # GT_train = r'D:\Monika\Proba_Data\GT'
    #
    # srcImagesVal = r'D:\Monika\Proba_Data\Validacija\Images'
    # GT_val = r'D:\Monika\Proba_Data\Validacija\GT'
    dstResultsPath = r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Results'
    dstModelsPath = r'D:\Monika\Models'
    dst_path_class_train = os.path.join(dstModelsPath, version, 'train_class.txt')
    dst_path_class_val = os.path.join(dstModelsPath, version, 'val_class.txt')
    dst_path_reg_train = os.path.join(dstModelsPath, version, 'train_reg.txt')
    dst_path_reg_val = os.path.join(dstModelsPath, version, 'train_val.txt')

    # gtDstPath = r'\\192.168.1.153\Science\Monika\GT'
    # resultspath=r'D:\Monika\PregledGT'
    # file_path_reg_coef = r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\reg_coef_cls_reg_cel_se'
    # --- create destination folders ---
    # if not os.path.exists(os.path.join(dstResultsPath, version)):
    #     os.mkdir(os.path.join(dstResultsPath, version))
    # else:
    #     # to avoid overwriting training results
    #     print(f"Folder name {version} exists.")
    #     exit(1)

    # resultsPath = os.path.join(dstResultsPath, version)
    #
    if not os.path.exists(os.path.join(dstModelsPath, version)):
        os.mkdir(os.path.join(dstModelsPath, version))

    modelsPath = os.path.join(dstModelsPath, version)
    if not os.path.exists(os.path.join(modelsPath, 'new')):
        os.mkdir(os.path.join(modelsPath, 'new'))
    modelsPath_new=os.path.join(modelsPath, 'new')
    # if not os.path.exists(os.path.join(dstModelsPath, version, 'reg_coef')):
    #     os.mkdir(os.path.join(dstModelsPath, version, 'reg_coef'))

    dst_path_class_train = os.path.join(dstModelsPath, version, 'train_class.txt')
    dst_path_class_val = os.path.join(dstModelsPath, version, 'val_class.txt')
    dst_path_reg_train = os.path.join(dstModelsPath, version, 'train_reg.txt')
    dst_path_reg_val = os.path.join(dstModelsPath, version, 'train_val.txt')

    # NOTE flags
    finetune = False

    # model = helper_model1.load_model(r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Models\vinf_1000_SO_DVA_INCEPTION_SKIP_48_normalizationWithMax_0.45_0.7\novi_bagnat\model.json', r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Models\vinf_1000_SO_DVA_INCEPTION_SKIP_48_normalizationWithMax_0.45_0.7\novi_bagnat\model.h5')

    # model_layer = model.get_layer()
    # weights_biases = model.get_weights()
    fixed_filters = False


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

    epochs = 3
    lr = 0.0001
    batch_size = 64
    im_size = (171, 256, 16)
    plot_color = (0, 0, 255)
    m=1
    filenames_GT=os.listdir(GT_train)
    filenames_val_GT=os.listdir(GT_val)

    # filenames_path_tmp = []
    # filenames_val_paths_tmp = []

    # for filename in filenames_GT:
    #     filenames_path_tmp.append(filename)
    #
    #
    # for filename in filenames:
    #     filenames_val_paths_tmp.append(os.path.join(dir, filename))

    filenames_paths=[]
    filenames_val_paths=[]

    indices = np.arange(len(filenames_GT))
    np.random.shuffle(indices)
    indices_val = np.arange(len(filenames_val_GT))
    np.random.shuffle(indices_val)

    for index in indices:
        filenames_paths.append(filenames_GT[index])

    for index in indices_val:
        filenames_val_paths.append(filenames_val_GT[index])

    for i in range(0,30):
        # print(filenames_paths)
        x_train=None
        out_class_train=None
        out_reg_train=None
        x_train, out_class_train, out_reg_train = helper_anchorless.get_image_gt_parallel_parts_renamed(GT_train,filenames_paths[i*int(len(filenames_paths)/30):(i+1)*int(len(filenames_paths)/30)], srcImagesPath, finetune)
        print('Training data loaded.')
        print(f'Shape of training images: {x_train.shape}')
        print(f'Shape of training classifier ground truth: {out_class_train.shape}')
        print(f'Shape of training regressor ground truth: {out_reg_train.shape}')




        x_val, out_class_val, out_reg_val = helper_anchorless.get_image_gt_parallel_parts_renamed(GT_val, filenames_val_paths[i*int(len(filenames_val_paths)/30):(i+1)*int(len(filenames_val_paths)/30)],srcImagesVal, finetune)
        print('Validation data loaded.')
        print(f'Shape of validation images: {x_train.shape}')
        print(f'Shape of validation classifier ground truth: {out_class_val.shape}')
        print(f'Shape of validation regressor ground truth: {out_reg_val.shape}')

        # print(x_train)
        print(out_class_train.shape)


        out_reg_norm = out_reg_train

        print(f'Koeficient pozicii: {reg_norm_coef_position_rows} {reg_norm_coef_position_cols}')
        print(f'Koeficient dimenzii: {reg_norm_coef_size_height} {reg_norm_coef_size_width}')

        out_reg_norm[:, :, :, 0] = out_reg_norm[:, :, :, 0] / reg_norm_coef_position_rows
        out_reg_norm[:, :, :, 1] = out_reg_norm[:, :, :, 1] / reg_norm_coef_position_cols

        out_reg_norm[:, :, :, 2] = out_reg_norm[:, :, :, 2] / reg_norm_coef_size_height
        out_reg_norm[:, :, :, 3] = out_reg_norm[:, :, :, 3] / reg_norm_coef_size_width


        # normalize validation data
        # out_reg_val_norm = deepcopy(out_reg_val)
        out_reg_val_norm = out_reg_val

        out_reg_val_norm[:, :, :, 0] = out_reg_val[:, :, :, 0] / reg_norm_coef_position_rows
        out_reg_val_norm[:, :, :, 1] = out_reg_val[:, :, :, 1] / reg_norm_coef_position_cols

        out_reg_val_norm[:, :, :, 2] = out_reg_val[:, :, :, 2] / reg_norm_coef_size_height
        out_reg_val_norm[:, :, :, 3] = out_reg_val[:, :, :, 3] / reg_norm_coef_size_width


        # --- specify number of GPUs to train - USER INPUT ---
        # use all available GPUs
        x = K.tensorflow_backend._get_available_gpus()
        G = len(x)

        # use a single GPU
        # G = 1   # user input
        if i==0:
            model = helper_model1.construct_model_anchorless_detector_wavelet_simple(input_shape=im_size)  # build model architecture
            # model_json = model.to_json()  # serialize model architecture to JSON
            # with open(os.path.join(os.path.join(modelsPath_new, 'model'+str(i)+'.json')), "w") as json_file:
            #     json_file.write(model_json)
        else:
            print(os.path.join(modelsPath,'model'+str(i-1)+'.json'))
            model=helper_model1.load_model(os.path.join(modelsPath_new,'model'+str(i-1)+'.json'), os.path.join(modelsPath_new,'model'+str(i-1)+'.h5'))

        if fixed_filters:
            filter = helper_model1.my_filter(shape=(3, 3, 1, 8), dtype=None)
            biases = np.zeros((8,))
            model.layers[1].set_weights([filter, biases])

            model.layers[1].trainable = False
            filter_2 = helper_model1.my_filter_2(shape=(3, 3, 32, 8), dtype=None)
            model.layers[4].set_weights([filter_2, biases])
            model.layers[4].trainable = False


        if G <= 1:
            print("[INFO] training with 1 GPU.")
        else:
            print("[INFO] training with {} GPUs.".format(G))
            model = multi_gpu_model(model, gpus=G)


        # import focal_loss
        # compile model

        model.compile(loss={
            'out_class': helper_losses.rpn_loss_cls_new,
            'out_reg': helper_losses.rpn_loss_reg

        },
            optimizer=Adam(lr=lr),
            metrics=['accuracy'])


        # --- fit model ---
        model_checkpoint = ModelCheckpoint(filepath=os.path.join(modelsPath, 'checkpoint-{epoch:03d}-{loss:.4f}.hdf5'),
                                           # epoch number and val accuracy will be part of the weight file name
                                           monitor='loss',  # metric to monitor when selecting weight checkpoints to save
                                           verbose=1,
                                           save_best_only=True)  # True saves only the weights after epochs where the monitored value (val accuracy) is improved

        plot_losses = helper_stats.TrainingPlot(dst_path_class_train, dst_path_class_val, dst_path_reg_train,
                                                dst_path_reg_val)  # enable live plot of training and validation loss that updates after every epoch, optional
        # add plot_losses to the list of callbacks in model.fit


        history = model.fit(x=x_train, y={"out_class": out_class_train, 'out_reg': out_reg_norm},
                            batch_size=batch_size,  # number of samples to process before updating the weights
                            epochs=epochs,
                            shuffle=True,
                            callbacks=[model_checkpoint, plot_losses],
                            verbose=1,
                            validation_data=(x_val, {"out_class": out_class_val, 'out_reg': out_reg_val_norm}))

        print('model fitted')



    # save model configuration and weights
    #     model_json = model.to_json()  # serialize model architecture to JSON
    #     with open(os.path.join(os.path.join(modelsPath, 'model.json')), "w") as json_file:
    #         json_file.write(model_json)
    #     model.save_weights(os.path.join(modelsPath, 'model.h5'))  # serialize weights to HDF5

        plot_model(model, to_file=os.path.join(modelsPath, 'modelDiagram.png'), show_shapes=True)  # save diagram of model architecture

        print('d')
        # --- save training curves and logs ---
        helper_stats.save_training_logs_ssd(history=history, dst_path=modelsPath)
        # m=i+1
        func_construct_model = helper_model1.construct_model_anchorless_detector_wavelet_simple
        # if i==0:
        #     m=i+1
        # else:
        #     m=m+2
        model = helper_model1.repack_as_single_gpu(model, func_construct_model, im_size, m,finetune=False)
        print(model.summary())


        # with open(os.path.join(modelsPath_new, 'modelSummary.txt'), 'w') as fh:  # save model summary
        #     model.summary(print_fn=lambda x: fh.write(x + '\n'))
            # plot_model(model, to_file=os.path.join(modelsPath, 'modelDiagram.png'), show_shapes=True)   # save diagram of model architecture

            # save model configuration and weights
        model_json = model.to_json()  # serialize model architecture to JSON
        with open(os.path.join(os.path.join(modelsPath_new, 'model'+str(i)+'.json')), "w") as json_file:
            json_file.write(model_json)
        model.save_weights(os.path.join(modelsPath_new, 'model'+str(i)+'.h5'))  # serialize weights to HDF5

        plot_model(model, to_file=os.path.join(modelsPath, 'modelDiagram.png'), show_shapes=True)  # save diagram of model architecture
        m+=2
        gc.collect()
# --- apply model to test data ---
# Y_test_pred = model.predict(x_test, verbose=1)
