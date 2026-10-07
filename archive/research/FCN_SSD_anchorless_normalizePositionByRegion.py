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
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
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






print(K.tensorflow_backend._get_available_gpus())  # check available number of gpus

if __name__ == '__main__':

    version = 'vinf_900_SO_DVA_INCEPTION_SKIP_48_normalizedByRegion_v4'
    # version = 'vinf_proba_ne'

    # --- paths ---
    srcImagesPath = r'D:\Monika\Trening_ZaDemo\Images'  # ova e eden folder
    # srcImagesPath = r'D:\Monika\Proba\Images'  # ova e eden folder
    # GT_train = r'D:\Monika\Proba\GT'
    GT_train = r'D:\Monika\Trening_ZaDemo\GT'

    srcImagesVal = r'D:\Monika\Trening_ZaDemo\Validacija\Images'
    # srcImagesVal = r'D:\Monika\Proba\Validacija\Images'
    GT_val = r'D:\Monika\Trening_ZaDemo\Validacija\GT'
    # GT_val = r'D:\Monika\Proba\Validacija\GT'

    dstResultsPath = r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Results'
    dstModelsPath = r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Models'
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

    if not os.path.exists(os.path.join(dstModelsPath, version, 'reg_coef')):
        os.mkdir(os.path.join(dstModelsPath, version, 'reg_coef'))
    file_path_reg_coef = os.path.join(dstModelsPath, version, 'reg_coef')

    dst_path_class_train = os.path.join(dstModelsPath, version, 'train_class.txt')
    dst_path_class_val = os.path.join(dstModelsPath, version, 'val_class.txt')
    dst_path_reg_train = os.path.join(dstModelsPath, version, 'train_reg.txt')
    dst_path_reg_val = os.path.join(dstModelsPath, version, 'train_val.txt')

    # --- variables ---
    imgDims = {'rows': 341, 'cols': 512}
    num_classes = 2
    img_depth = 1

    img_dims = (imgDims['rows'], imgDims['cols'], img_depth)

    # x_train, out_class_train, out_reg_train = helper_anchorless.get_image_gt(GT_train, srcImagesPath)
    # print("procitani trening")

    x_train, out_class_train, out_reg_train = helper_anchorless.get_image_gt_parallel(GT_train, srcImagesPath)
    print('Training data loaded.')
    print(f'Shape of training images: {x_train.shape}')
    print(f'Shape of training classifier ground truth: {out_class_train.shape}')
    print(f'Shape of training regressor ground truth: {out_reg_train.shape}')

    # x_val, out_class_val, out_reg_val = helper_anchorless.get_image_gt(GT_val, srcImagesVal)
    # # x_test, bboxes_test = helper_data.read_data_rpn(os.path.join(srcImagesPath, 'test'), (imgDims['cols'], imgDims['rows']), img_depth, srcAnnotationsPath, exclude_empty=False)

    x_val, out_class_val, out_reg_val = helper_anchorless.get_image_gt_parallel(GT_val, srcImagesVal)
    print('Validation data loaded.')
    print(f'Shape of validation images: {x_train.shape}')
    print(f'Shape of validation classifier ground truth: {out_class_val.shape}')
    print(f'Shape of validation regressor ground truth: {out_reg_val.shape}')

    # print(x_train)
    print(out_class_train.shape)

    anchor_stride = 8
    # norm_coef = 100     # constant to normalize regression ground truth data

    # NOTE: V3 - each output separately, normalized relative to maximum annotation size in image region
    #
    # # position
    # reg_norm_coef_position_rows = np.max(np.abs(out_reg_train[:, :, :, 0]))
    # filename = 'reg_norm_coef_position_rows.txt'
    # fid1_1 = open(os.path.join(file_path_reg_coef, filename), 'wb+')
    # pickle.dump(reg_norm_coef_position_rows, fid1_1)
    # fid1_1.close()
    #
    # reg_norm_coef_position_cols = np.max(np.abs(out_reg_train[:, :, :, 1]))
    # filename = 'reg_norm_coef_position_cols.txt'
    # fid1_1 = open(os.path.join(file_path_reg_coef, filename), 'wb+')
    # pickle.dump(reg_norm_coef_position_cols, fid1_1)
    # fid1_1.close()
    #
    # # size
    # reg_norm_coef_size_height = np.max(np.abs(out_reg_train[:, :, :, 2]))
    # filename = 'reg_norm_coef_size_height.txt'
    # fid1_2 = open(os.path.join(file_path_reg_coef, filename), 'wb+')
    # pickle.dump(reg_norm_coef_size_height, fid1_2)
    # fid1_2.close()
    #
    # reg_norm_coef_size_width = np.max(np.abs(out_reg_train[:, :, :, 3]))
    # filename = 'reg_norm_coef_size_width.txt'
    # fid1_2 = open(os.path.join(file_path_reg_coef, filename), 'wb+')
    # pickle.dump(reg_norm_coef_size_width, fid1_2)
    # fid1_2.close()

    # normalize training data
    # out_reg_norm = deepcopy(out_reg_train)    # NOTE: EV - ne mi teknuva oti sme troshele duplo memorija, nadole ne se koristat originalnite otkoga se premesti crtanjeto vo druga skripta
    out_reg_norm = out_reg_train

    out_reg_val_norm = out_reg_val

    # print(f'Max od pozicii - trening PRED: {np.max(np.abs(out_reg_norm[:, :, :, 0:2]))}')
    # print(f'Max od dimenzii - trening PRED: {np.max(np.abs(out_reg_norm[:, :, :, 2:]))}')
    # print(f'Min od pozicii - trening PRED: {np.min(np.abs(out_reg_norm[:, :, :, 0:2]))}')
    # print(f'Min od dimenzii - trening PRED: {np.min(np.abs(out_reg_norm[:, :, :, 2:]))}')

    # print(f'Koeficient pozicii: {reg_norm_coef_position_rows} {reg_norm_coef_position_cols}')
    # print(f'Koeficient dimenzii: {reg_norm_coef_size_height} {reg_norm_coef_size_width}')


    # pozicija redici
    regions = [0, 50, 120, 270, 341]   # regioni po redici (centroidi), vo resizuvana slika (ne vo kvantiziran prostor)
    norm_coefs = [15, 40, 50, 63]
    destination_matrix = 0  # pomestuvanje redici

    out_reg_norm, out_reg_val_norm = helper_anchorless.normalize_output_by_region(
        regions, norm_coefs, destination_matrix, out_reg_norm, out_reg_val_norm, anchor_stride)

    f = open(os.path.join(file_path_reg_coef,'reg_norm_coeff.txt'), 'a+')
    f.write(str(regions))
    f.write("\n")
    f.write(str(norm_coefs))
    f.write("\n")
    f.close()


    # pozicija koloni
    regions = [0, 50, 120, 190, 341]
    norm_coefs = [18, 37, 70, 85]   # [10, 27, 45, 60]

    destination_matrix = 1  # pomestuvanje koloni

    out_reg_norm, out_reg_val_norm = helper_anchorless.normalize_output_by_region(
        regions, norm_coefs, destination_matrix, out_reg_norm, out_reg_val_norm, anchor_stride)

    f = open(os.path.join(file_path_reg_coef,'reg_norm_coeff.txt'), 'a+')
    f.write(str(regions))
    f.write("\n")
    f.write(str(norm_coefs))
    f.write("\n")
    f.close()

    # dimenzija visina
    regions = [0, 20, 50, 75, 100, 150, 341]
    norm_coefs = [25, 50, 75, 110, 150, 210]   # vo celata vlezna slika
    norm_coefs = [float(x)/img_dims[0] for x in norm_coefs]   # vo kvantiziraniot izlez

    destination_matrix = 2  # visina

    out_reg_norm, out_reg_val_norm = helper_anchorless.normalize_output_by_region(
        regions, norm_coefs, destination_matrix, out_reg_norm, out_reg_val_norm, anchor_stride)

    f = open(os.path.join(file_path_reg_coef,'reg_norm_coeff.txt'), 'a+')
    f.write(str(regions))
    f.write("\n")
    f.write(str(norm_coefs))
    f.write("\n")
    f.close()

    # dimenzija shirina
    regions = [0, 50, 100, 170, 341]
    norm_coefs = [60, 100, 160, 190]    # vo celata vlezna slika
    norm_coefs = [float(x / img_dims[1]) for x in norm_coefs]  # vo kvantiziraniot izlez

    destination_matrix = 3  # shirina

    out_reg_norm, out_reg_val_norm = helper_anchorless.normalize_output_by_region(
        regions, norm_coefs, destination_matrix, out_reg_norm, out_reg_val_norm, anchor_stride)

    f = open(os.path.join(file_path_reg_coef,'reg_norm_coeff.txt'), 'a+')
    f.write(str(regions))
    f.write("\n")
    f.write(str(norm_coefs))
    f.close()


    print(f'Max od pozicii - trening: {np.max(np.abs(out_reg_norm[:, :, :, 0:2]))}')
    print(f'Max od dimenzii - trening: {np.max(np.abs(out_reg_norm[:, :, :, 2:]))}')
    print(f'Min od pozicii - trening: {np.min(np.abs(out_reg_norm[:, :, :, 0:2]))}')
    print(f'Min od dimenzii - trening: {np.min(np.abs(out_reg_norm[:, :, :, 2:]))}')

    print(f'Max od pozicii - validacija: {np.max(np.abs(out_reg_val_norm[:, :, :, 0:2]))}')
    print(f'Max od dimenzii - validacija: {np.max(np.abs(out_reg_val_norm[:, :, :, 2:]))}')

    # exit(1)

    # optimization hyperprameters
    epochs = 30
    lr = 0.0001
    batch_size = 64
    im_size = (341, 512, 1)
    plot_color = (0, 0, 255)
    prob_thr = 0.5

    # anchorless_genertor_plot.save_results_anchorless_limits_cls(resultspath, x_train, out_class_train, out_reg_train, anchor_stride, prob_thr)
    model = helper_model1.construct_model_anchorless_detector_skip_v1(input_shape=im_size)  # build model architecture
    #   model = helper_model1.load_model(r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Models\vinf_800_SO_DVA_INCEPTION\new\model.json', r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Models\vinf_800_SO_DVA_INCEPTION\new\model.h5')

    # --- specify number of GPUs to train - USER INPUT ---
    # use all available GPUs
    x = K.tensorflow_backend._get_available_gpus()
    G = len(x)

    # use a single GPU
    # G = 1   # user input

    if G <= 1:
        print("[INFO] training with 1 GPU.")
    else:
        print("[INFO] training with {} GPUs.".format(G))
        model = multi_gpu_model(model, gpus=G)

    model_json = model.to_json()  # serialize model architecture to JSON
    with open(os.path.join(os.path.join(modelsPath, 'model.json')), "w") as json_file:
        json_file.write(model_json)

    # compile model
    model.compile(loss={
        'out_class': helper_losses.rpn_loss_cls,
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

    # history = model.fit(X_train, Y_train,
    #                     batch_size=batch_size,  # number of samples to process before updating the weights
    #                     epochs=epochs,
    #                     callbacks=[model_checkpoint, plot_losses],
    #                     verbose=1,
    #                     validation_data=(X_val, Y_val))

    history = model.fit(x=x_train, y={"out_class": out_class_train, 'out_reg': out_reg_norm},
                        batch_size=batch_size,  # number of samples to process before updating the weights
                        epochs=epochs,
                        callbacks=[model_checkpoint, plot_losses],

                        verbose=1,
                        validation_data=(x_val, {"out_class": out_class_val, 'out_reg': out_reg_val_norm}))

    print('model fitted')

    # --- save model ---
    # save model architecture
    # repack model as single-gpu
    # func_construct_model = helper_model1.construct_model_anchorless_detector
    # model = helper_model1.repack_as_single_gpu(model, func_construct_model, im_size)

    print(model.summary())  # parameter info for each layer
    with open(os.path.join(modelsPath, 'modelSummary.txt'), 'w') as fh:  # save model summary
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    # plot_model(model - out_class_loss: 0.0321 - out_reg_loss: 2.0330e-04 - out_class_accuracy: 0.0315 - out_reg_accuracy: 0.0121

    # save model configuration and weights
    model_json = model.to_json()  # serialize model architecture to JSON
    with open(os.path.join(os.path.join(modelsPath, 'model.json')), "w") as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(modelsPath, 'model.h5'))  # serialize weights to HDF5

    plot_model(model, to_file=os.path.join(modelsPath, 'modelDiagram.png'), show_shapes=True)  # save diagram of model architecture

    print('d')
    # --- save training curves and logs ---
    helper_stats.save_training_logs_ssd(history=history, dst_path=modelsPath)

    func_construct_model = helper_model1.construct_model_anchorless_detector_skip_v1
    model = helper_model1.repack_as_single_gpu(model, func_construct_model, im_size, finetune=False)
    print(model.summary())
    with open(os.path.join(modelsPath, 'new', 'modelSummary.txt'), 'w') as fh:  # save model summary
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
        # plot_model(model, to_file=os.path.join(modelsPath, 'modelDiagram.png'), show_shapes=True)   # save diagram of model architecture

        # save model configuration and weights
    model_json = model.to_json()  # serialize model architecture to JSON
    with open(os.path.join(os.path.join(modelsPath, 'new', 'model.json')), "w") as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(modelsPath, 'new', 'model.h5'))  # serialize weights to HDF5

    plot_model(model, to_file=os.path.join(modelsPath, 'new', 'modelDiagram.png'), show_shapes=True)  # save diagram of model architecture

    # --- apply model to test data ---
    # Y_test_pred = model.predict(x_test, verbose=1)
