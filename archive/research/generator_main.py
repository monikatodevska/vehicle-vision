
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

import h5py
# custom package imports
#from SingleShotDetector.Helpers import helper_model, helper_data, helper_stats, losses
import sys
#sys.path.insert(0, 'E:\Science\Monika\SingleShotDetector\Helpers' )
#import cv2
import generator
import helper_model, helper_stats, helper_losses
import helper_data
import cv2
# --- paths ---
if __name__ == '__main__':
    version = 'vinf_12'

    srcImagesPath = r'E:\Science\Monika'
    srcImagesPathTrain = r'E:\Science\Monika\train1'
    srcImagesPathVal = r'E:\Science\Monika\val1'
    srcAnnotationsPath = r'E:\Science\Monika\GRAM-RTMv4\Annotations\site2'
    srcAnnotationsPathTrain = r'E:\Science\Monika\GRAM-RTMv4\Annotations\train-annot'
    dstResultsPath = r'E:\Science\Monika\Results'
    dstModelsPath = r'E:\Science\Monika\Models'
    gtDstPath = r'E:\Science\Monika\GT'

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
    imgDims = {'rows': 480, 'cols': 800}

    num_classes = 1
    img_depth = 1
    im_size=[480, 800,1]

    anchor_dims = ((25, 25), (32, 32), (48, 48), (64, 64), (92, 92))
    anchor_stride = 8
    norm_coef = 100  # constant to normalize regression ground truth data

    # IOU thresholds for selecting positive and negative anchors
    iou_low = 0.4
    iou_high = 0.6

    train_idx_p = os.listdir(srcImagesPathTrain)
    train_idx = []
    for i, im_name in enumerate(train_idx_p):
        index = int(im_name[5:11])
        train_idx.append(index)

    val_idx_p = os.listdir(srcImagesPathVal)
    val_idx = []
    for i, im_name in enumerate(val_idx_p):
        index = int(im_name[5:11])
        val_idx.append(index)

    print(1)

    training_generator = generator.DataGenerator(train_idx, os.path.join(srcImagesPath, 'train1'), srcAnnotationsPathTrain,  anchor_stride, anchor_dims, iou_low, iou_high, im_size, img_depth, batch_size=32, exclude_empty=False, shuffle=False, to_fit=True, n_classes=1)

    validation_generator = generator.DataGenerator(val_idx, os.path.join(srcImagesPath, 'val1'), srcAnnotationsPath,  anchor_stride, anchor_dims, iou_low, iou_high, im_size, img_depth,  batch_size=32, exclude_empty=False, shuffle=False, to_fit=True, n_classes=1)

    epochs = 30
    lr = 0.0001
    batch_size = 40

    model = helper_model.construct_model_ssd_cls(input_shape=im_size, num_anchors=len(anchor_dims))  # build model architecture

    # compile model
    model.compile(loss={
        'out_class': helper_losses.rpn_loss_cls

    },
        optimizer=Adam(lr=lr),
        metrics=['accuracy'])

    # --- fit model ---
    model_checkpoint = ModelCheckpoint(filepath=os.path.join(modelsPath, 'checkpoint-{epoch:03d}-{loss:.4f}.hdf5'),  # epoch number and val accuracy will be part of the weight file name
                                       monitor='loss',  # metric to monitor when selecting weight checkpoints to save
                                       verbose=1,
                                       save_best_only=True)  # True saves only the weights after epochs where the monitored value (val accuracy) is improved

    history = model.fit_generator(training_generator,
                        # steps_per_epoch=len(train_idx)/batch_size,
                        callbacks=[model_checkpoint],
                        verbose=1,
                        validation_data=validation_generator
                        )

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

    print('d')
    # --- save training curves and logs ---
    helper_stats.save_training_logs_ssd(history=history, dst_path=modelsPath)