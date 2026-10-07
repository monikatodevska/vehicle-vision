
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

print(K.tensorflow_backend._get_available_gpus())   # check available number of gpus


if __name__ == '__main__':

    version = 'vinf_201_clsOnly'
    # version = 'vinf_proba_ne'

    # --- paths ---
    srcImagesPath = r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Complete_cls\AugmentedPhotos1'  # ova e eden folder
    GT_train = r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Complete_cls\AugmentedGT1'
    GT_val = r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Complete_cls\Validacija\GT_Val' #da se namesti koga ke se sredat za validacija
    dstResultsPath = r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Results'
    dstModelsPath = r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Models'
    # gtDstPath = r'\\192.168.1.153\Science\Monika\GT'
    srcImagesVal = r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Complete_cls\Validacija\Images_Val'
    file_path_reg_coef = r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\reg_coef_clsOnly'
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

    # --- variables ---
    imgDims = {'rows': 341, 'cols': 512}
    num_classes = 2
    img_depth = 1

    img_dims = (imgDims['rows'], imgDims['cols'], img_depth)

    # --- load and format data ---

    # x_train, out_class_train, out_reg_train = helper_anchorless.get_image_gt(GT_train, srcImagesPath)
    # print("procitani trening")

    x_train, out_class_train = helper_anchorless.get_image_gt_cls_parallel(GT_train, srcImagesPath)
    print('Training data loaded.')
    print(f'Shape of training images: {x_train.shape}')
    print(f'Shape of training classifier ground truth: {out_class_train.shape}')

    # x_val, out_class_val, out_reg_val = helper_anchorless.get_image_gt(GT_val, srcImagesVal)
    # # x_test, bboxes_test = helper_data.read_data_rpn(os.path.join(srcImagesPath, 'test'), (imgDims['cols'], imgDims['rows']), img_depth, srcAnnotationsPath, exclude_empty=False)

    x_val, out_class_val = helper_anchorless.get_image_gt_cls_parallel(GT_val, srcImagesVal)
    print('Validation data loaded.')
    print(f'Shape of validation images: {x_train.shape}')
    print(f'Shape of validation classifier ground truth: {out_class_val.shape}')

    # print(x_train)
    print(out_class_train.shape)

    anchor_stride = 8

    # optimization hyperprameters
    epochs = 25
    lr = 0.0001
    batch_size = 128
    im_size = [341, 512, 1]

    model = helper_model1.construct_model_anchorless_detector_cls(input_shape=im_size)  # build model architecture
    # model = helper_model1.load_model(r'E:\Science\Monika\Models\vinf_156_dotreniranje\model.json', r'E:\Science\Monika\Models\vinf_156_dotreniranje\model.h5')

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

    # compile model
    model.compile(loss={
        'out_class': helper_losses.rpn_loss_cls,
    },
        optimizer=Adam(lr=lr),
        metrics=['accuracy'])

    # --- fit model ---
    model_checkpoint = ModelCheckpoint(filepath=os.path.join(modelsPath, 'checkpoint-{epoch:03d}-{loss:.4f}.hdf5'),
                                       # epoch number and val accuracy will be part of the weight file name
                                       monitor='loss',  # metric to monitor when selecting weight checkpoints to save
                                       verbose=1,
                                       save_best_only=True)  # True saves only the weights after epochs where the monitored value (val accuracy) is improved

    history = model.fit(x=x_train, y={"out_class": out_class_train},
                        batch_size=batch_size,  # number of samples to process before updating the weights
                        epochs=epochs,
                        callbacks=[model_checkpoint],
                        verbose=1,
                        validation_data=(x_val, {"out_class": out_class_val}))

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
    helper_stats.save_training_logs(history=history, dst_path=modelsPath)

    # --- apply model to test data ---
    # Y_test_pred = model.predict(x_test, verbose=1)
