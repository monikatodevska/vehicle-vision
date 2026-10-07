"""
Course: Mashinski vid, FEEIT, Spring 2021
Date: 09.08.2021

Description: design, train and evaluate a fully convolutional SSD architecture for object classification and localization
Python version: 3.6
"""

# python imports
import os
import pickle
from copy import deepcopy

import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.utils.multi_gpu_utils import multi_gpu_model

import helper_anchorless
import helper_losses
import helper_model1
import helper_stats

# custom package imports
# from SingleShotDetector.Helpers import helper_model, helper_data, helper_stats, losses
# sys.path.insert(0, 'E:\Science\Monika\SingleShotDetector\Helpers' )
# import cv2

print(K.tensorflow_backend._get_available_gpus())  # check available number of gpus

if __name__ == '__main__':

    # version = 'vinf_200_finetuned1_mil_cel'
    # version = r'vinf_1100_SO_DVA_INCEPTION_SKIP_48_normalizationWithMax_0.45_0.7_trening_pak1_finetune'
    version = r'v5000_0.6_0.7_Trening_With_Translations-Up-Down-All_Cams_softmax_CUSTOM_CATEGORICAL_LOSS_AXES__10^-1_reg_coef_FINETUNE_CEL_DOPOLNETI_PRAVILEN'

    # --- paths ---
    srcImagesPath = r'D:\Monika\Trening_With_Translations-Up-Down-All_Cams_NEW1\Images'  # ova e eden folder
    GT_train = r'D:\Monika\Trening_With_Translations-Up-Down-All_Cams_NEW1\GT'

    #model
    # dstResultsPath = r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Results'
    dstModelsPath = r'D:\Monika\Models'
    # gtDstPath = r'\\192.168.1.153\Science\Monika\GT'
    #validation data
    GT_val = r'D:\Monika\Trening_With_Translations-Up-Down-All_Cams_NEW1\Validacija\GT'
    srcImagesVal = r'D:\Monika\Trening_With_Translations-Up-Down-All_Cams_NEW1\Validacija\Images'
    resultspath = r'D:\Monika\PregledGT'

    # srcImagesPath = r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Dotreniranje\AugmentedPhotos'  # ova e eden folder
    # GT_train = r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Dotreniranje\AugmentedGT'
    # GT_val = r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Dotreniranje\Validacija\GT_Val'
    # # dstResultsPath = r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Results'
    # dstModelsPath = r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Models'
    # # gtDstPath = r'\\192.168.1.153\Science\Monika\GT'
    # srcImagesVal = r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Dotreniranje\Validacija\Images_Val'
    # file_path_reg_coef = r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles'
    # file_path_reg_coef = r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\reg_coef_finetune_cel'
    # file_path_reg_coef_old = r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\reg_coef'
    # --- create destination folders ---
    # if not os.path.exists(os.path.join(dstResultsPath, version)):
    #     os.mkdir(os.path.join(dstResultsPath, version))
    # else:
    #     # to avoid overwriting training results
    #     print(f"Folder name {version} exists.")
    #     exit(1)

    # resultsPath = os.path.join(dstResultsPath, version)

    if not os.path.exists(os.path.join(dstModelsPath, version)):
        os.mkdir(os.path.join(dstModelsPath, version))
    modelsPath = os.path.join(dstModelsPath, version)

    if not os.path.exists(os.path.join(dstModelsPath, version, 'reg_coef')):
        os.mkdir(os.path.join(dstModelsPath, version, 'reg_coef'))
    file_path_reg_coef = os.path.join(dstModelsPath, version, 'reg_coef')

    dst_path_class_train = os.path.join(dstModelsPath, version, 'train_class.txt')
    dst_path_class_val = os.path.join(dstModelsPath, version, 'val_class.txt')
    dst_path_reg_train = os.path.join(dstModelsPath, version, 'train_reg.txt')
    dst_path_reg_val = os.path.join(dstModelsPath, version, 'val_reg.txt')
    # OriginalModelsPath = r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Models\vinf_1000_SO_DVA_INCEPTION_SKIP_48_normalizationWithMax_0.45_0.7_bezKineski_bezPolaM30/new'
    # OriginalModelsPath = r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Models\vinf_1100_SO_DVA_INCEPTION_SKIP_48_normalizationWithMax_0.45_0.7_trening_pak1'
    OriginalModelsPath = r'D:\Monika\Models\v5000_0.6_0.7_Trening_With_Translations-Up-Down-All_Cams_softmax_CUSTOM_CATEGORICAL_LOSS_AXES__10^-1_reg_coef\new'

    # --- flags ---
    flag_train_only_regressor =False
    finetune=True
    # --- variables ---
    imgDims = {'rows': 341, 'cols': 512}
    num_classes = 2
    img_depth = 1

    img_dims = (imgDims['rows'], imgDims['cols'], img_depth)

    # --- load and format data ---

    # x_train, out_class_train, out_reg_train = helper_anchorless.get_image_gt(GT_train, srcImagesPath)
    # print("procitani trening")

    #NOTE da se smeni vo funkcijava koga se trenira a ne se dotrenira da gi zema site folderi( sega se frlaat kineski i m-30)
    x_train, out_class_train, out_reg_train = helper_anchorless.get_image_gt_parallel(GT_train, srcImagesPath, finetune)
    print('Training data loaded.')
    print(f'Shape of training images: {x_train.shape}')
    print(f'Shape of training classifier ground truth: {out_class_train.shape}')
    print(f'Shape of training regressor ground truth: {out_reg_train.shape}')

    # x_val, out_class_val, out_reg_val = helper_anchorless.get_image_gt(GT_val, srcImagesVal)

    x_val, out_class_val, out_reg_val = helper_anchorless.get_image_gt_parallel(GT_val, srcImagesVal,finetune)
    print('Validation data loaded.')
    print(f'Shape of validation images: {x_train.shape}')
    print(f'Shape of validation classifier ground truth: {out_class_val.shape}')
    print(f'Shape of validation regressor ground truth: {out_reg_val.shape}')

    # print(x_train)
    print(out_class_train.shape)

    anchor_stride = 8
    # norm_coef = 100     # constant to normalize regression ground truth data

    # IOU thresholds for selecting positive and negative anchors
    # iou_low = 0.4
    # iou_high = 0.6

    # reg_norm_coef_position = np.max(np.abs(out_reg_train[:, :, :, 0:2]))
    # filename = 'reg_norm_coef_position.txt'
    # fid1_1 = open(os.path.join(file_path_reg_coef, filename), 'wb+')
    # pickle.dump(reg_norm_coef_position, fid1_1)
    # fid1_1.close()
    #
    # reg_norm_coef_size = np.max(np.abs(out_reg_train[:, :, :, 2:]))
    # filename = 'reg_norm_coef_size.txt'
    # fid1_2 = open(os.path.join(file_path_reg_coef, filename), 'wb+')
    # pickle.dump(reg_norm_coef_size, fid1_2)
    # fid1_2.close()

    '''
    filename = 'reg_norm_coef_position.txt'
    fid1_1 = open(os.path.join(file_path_reg_coef_old, filename), 'rb')
    reg_norm_coef_position = pickle.load(fid1_1)
    fid1_1.close()

    filename = 'reg_norm_coef_size.txt'
    fid1_2 = open(os.path.join(file_path_reg_coef_old, filename), 'rb')
    reg_norm_coef_size = pickle.load(fid1_2)
    fid1_2.close()
    '''


    # position
    reg_norm_coef_position_rows = np.max(np.abs(out_reg_train[:, :, :, 0]))
    filename = 'reg_norm_coef_position_rows.txt'
    fid1_1 = open(os.path.join(file_path_reg_coef, filename), 'wb+')
    pickle.dump(reg_norm_coef_position_rows, fid1_1)
    fid1_1.close()

    reg_norm_coef_position_cols = np.max(np.abs(out_reg_train[:, :, :, 1]))
    filename = 'reg_norm_coef_position_cols.txt'
    fid1_1 = open(os.path.join(file_path_reg_coef, filename), 'wb+')
    pickle.dump(reg_norm_coef_position_cols, fid1_1)
    fid1_1.close()

    # size
    reg_norm_coef_size_height = np.max(np.abs(out_reg_train[:, :, :, 2]))
    filename = 'reg_norm_coef_size_height.txt'
    fid1_2 = open(os.path.join(file_path_reg_coef, filename), 'wb+')
    pickle.dump(reg_norm_coef_size_height, fid1_2)
    fid1_2.close()

    reg_norm_coef_size_width = np.max(np.abs(out_reg_train[:, :, :, 3]))
    filename = 'reg_norm_coef_size_width.txt'
    fid1_2 = open(os.path.join(file_path_reg_coef, filename), 'wb+')
    pickle.dump(reg_norm_coef_size_width, fid1_2)
    fid1_2.close()

    # normalize training data
    # out_reg_norm = deepcopy(out_reg_train)    # NOTE: EV - ne mi teknuva oti sme troshele duplo memorija, nadole ne se koristat originalnite otkoga se premesti crtanjeto vo druga skripta
    out_reg_norm = out_reg_train

    # print(f'Max od pozicii - trening PRED: {np.max(np.abs(out_reg_norm[:, :, :, 0:2]))}')
    # print(f'Max od dimenzii - trening PRED: {np.max(np.abs(out_reg_norm[:, :, :, 2:]))}')
    # print(f'Min od pozicii - trening PRED: {np.min(np.abs(out_reg_norm[:, :, :, 0:2]))}')
    # print(f'Min od dimenzii - trening PRED: {np.min(np.abs(out_reg_norm[:, :, :, 2:]))}')

    print(f'Koeficient pozicii: {reg_norm_coef_position_rows} {reg_norm_coef_position_cols}')
    print(f'Koeficient dimenzii: {reg_norm_coef_size_height} {reg_norm_coef_size_width}')

    out_reg_norm[:, :, :, 0] = out_reg_norm[:, :, :, 0] / reg_norm_coef_position_rows
    out_reg_norm[:, :, :, 1] = out_reg_norm[:, :, :, 1] / reg_norm_coef_position_cols

    out_reg_norm[:, :, :, 2] = out_reg_norm[:, :, :, 2] / reg_norm_coef_size_height
    out_reg_norm[:, :, :, 3] = out_reg_norm[:, :, :, 3] / reg_norm_coef_size_width

    # print(f'Max od pozicii - trening: {np.max(np.abs(out_reg_norm[:, :, :, 0:2]))}')
    # print(f'Max od dimenzii - trening: {np.max(np.abs(out_reg_norm[:, :, :, 2:]))}')
    # print(f'Min od pozicii - trening: {np.min(np.abs(out_reg_norm[:, :, :, 0:2]))}')
    # print(f'Min od dimenzii - trening: {np.min(np.abs(out_reg_norm[:, :, :, 2:]))}')

    # normalize validation data
    # out_reg_val_norm = deepcopy(out_reg_val)
    out_reg_val_norm = out_reg_val

    out_reg_val_norm[:, :, :, 0] = out_reg_val[:, :, :, 0] / reg_norm_coef_position_rows
    out_reg_val_norm[:, :, :, 1] = out_reg_val[:, :, :, 1] / reg_norm_coef_position_cols

    out_reg_val_norm[:, :, :, 2] = out_reg_val[:, :, :, 2] / reg_norm_coef_size_height
    out_reg_val_norm[:, :, :, 3] = out_reg_val[:, :, :, 3] / reg_norm_coef_size_width

    # # normalize training data
    # out_reg_norm = deepcopy(out_reg_train)
    # out_reg_norm[:, :, :, 0:2] = out_reg_train[:, :, :, 0:2] / reg_norm_coef_position
    # out_reg_norm[:, :, :, 2:] = out_reg_train[:, :, :, 2:] / reg_norm_coef_size
    #
    # print(f'Max od pozicii - trening: {np.max(np.abs(out_reg_norm[:, :, :, 0:2]))}')
    # print(f'Max od dimenzii - trening: {np.max(np.abs(out_reg_norm[:, :, :, 2:]))}')
    #
    # # normalize validation data
    # out_reg_val_norm = deepcopy(out_reg_val)
    # out_reg_val_norm[:, :, :, 0:2] = out_reg_val[:, :, :, 0:2] / reg_norm_coef_position
    # out_reg_val_norm[:, :, :, 2:] = out_reg_val[:, :, :, 2:] / reg_norm_coef_size
    #
    # print(f'Max od pozicii - validacija: {np.max(np.abs(out_reg_val_norm[:, :, :, 0:2]))}')
    # print(f'Max od dimenzii - validacija: {np.max(np.abs(out_reg_val_norm[:, :, :, 2:]))}')

    # optimization hyperprameters
    epochs = 25
    lr = 0.00005
    batch_size = 64
    im_size = (341, 512, 1)

    # load model and weights
    model = helper_model1.load_model(model_path=os.path.join(OriginalModelsPath,'model.json'),
                                     weights_path=os.path.join(OriginalModelsPath,'model.h5'))

    print(model.summary())  # to find indices of model layers

    if flag_train_only_regressor:
        # freeze layers of feature extractor and classifier
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22]:   # NOTE: da se navedat sloevite koi nema da se treniraat
            model.layers[i].trainable = False

        # print value of trainable parameter
        for layer in model.layers:
            print(layer, layer.trainable)

    # --- specify number of GPUs to train - USER INPUT ---
    # use all available GPUs
    x = K.tensorflow_backend._get_available_gpus()
    G = len(x)

    # use a single GPU
    # G = 1   # user inpu

    if G <= 1:
         print("[INFO] training with 1 GPU.")
    else:
         print("[INFO] training with {} GPUs.".format(G))
         model = multi_gpu_model(model, gpus=G)

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

    plot_losses = helper_stats.TrainingPlot(dst_path_class_train, dst_path_class_val,dst_path_reg_train, dst_path_reg_val)  # enable live plot of training and validation loss that updates after every epoch, optional

    history = model.fit(x=x_train, y={"out_class": out_class_train, 'out_reg': out_reg_norm},
                        batch_size=batch_size,  # number of samples to process before updating the weights
                        epochs=epochs,
                        shuffle=True,
                        callbacks=[model_checkpoint, plot_losses],
                        verbose=1,
                        validation_data=(x_val, {"out_class": out_class_val, 'out_reg': out_reg_val_norm}))

    print('model fitted')



    # --- save model ---
    # save model architecture
    print(model.summary())  # parameter info for each layer
    with open(os.path.join(modelsPath, 'modelSummary.txt'), 'w') as fh:  # save model summary
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    # plot_model(model, to_file=os.path.join(modelsPath, 'modelDiagram.png'), show_shapes=True)   # save diagram of model architecture

    # save model configuration and weights
    model_json = model.to_json()  # serialize model architecture to JSON
    with open(os.path.join(os.path.join(modelsPath, 'model.json')), "w") as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(modelsPath, 'model.h5'))  # serialize weights to HDF5

    plot_model(model, to_file=os.path.join(modelsPath, 'modelDiagram.png'), show_shapes=True)  # save diagram of model architecture

    print('d')
    # --- save training curves and logs ---
    helper_stats.save_training_logs_ssd(history=history, dst_path=modelsPath)

    # repack model as single-gpu
    func_construct_model = helper_model1.construct_model_anchorless_detector_skip_v1
    model = helper_model1.repack_as_single_gpu(model, func_construct_model, im_size,finetune=True)

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
