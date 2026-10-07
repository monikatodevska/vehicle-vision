"""
Course: Mashinski vid, FEEIT, Spring 2022
Date: 28.03.2022

Description: design and train an anchorless fully convolutional network
             for single-shot object classification and localization
Python version: 3.6
"""

# python imports
import os
import numpy as np
from tqdm import tqdm

from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from joblib import Parallel, delayed

# custom package imports
# from Helpers_Anchorless import helper_model
# from Helpers_Anchorless import helper_data
# from Helpers_Anchorless import helper_stats
# from Helpers_Anchorless import helper_losses


if __name__ == '__main__':

    # --- paths ---
    version = 'Anchorless_tmp_6'

    srcImagesPath = r'D:\Science\Elena\MachineVision\Data\M-30\images_split'
    srcAnnotationsPath = r'D:\Science\Elena\MachineVision\Data\M-30\GRAM-RTMv4\Annotations\M-30\xml'

    dstModelsPath = r'D:\Science\Elena\MachineVision\Models_tmp'
    gtDstPath = r'D:\Science\Elena\MachineVision\GT_anchorless_cls_2'

    # create folders to save data from the current execution
    modelsPath = os.path.join(dstModelsPath, version)
    if not os.path.exists(modelsPath):
        os.mkdir(modelsPath)
    else:
        # to avoid overwriting training results
        print(f"Folder name {version} exists.")
        exit(1)

    # --- variables ---
    imgDims = {'rows': 480, 'cols': 800}  # input image dimensions
    num_classes = 1
    img_depth = 1  # 1 - grayscale, 3 - color
    img_dims = (imgDims['rows'], imgDims['cols'], img_depth)

    min_obj_height = 20  # minimum object height in pixels

    # --- load and format data ---
    # load full dataset into memory - image data and ground truth bounding boxes
    x_train_orig, bboxes_train = helper_data.read_data_rpn(os.path.join(srcImagesPath, 'train'),
                                                           (imgDims['cols'], imgDims['rows']), img_depth,
                                                           srcAnnotationsPath,
                                                           min_obj_height, exclude_empty=True, shuffle=False)

    x_val_orig, bboxes_val = helper_data.read_data_rpn(os.path.join(srcImagesPath, 'val'),
                                                       (imgDims['cols'], imgDims['rows']), img_depth,
                                                       srcAnnotationsPath,
                                                       min_obj_height, exclude_empty=True, shuffle=False)

    print(f'Training dataset shape: {x_train_orig.shape}')
    print(f'Number of training samples: {x_train_orig.shape[0]}')
    print(f'Number of validation samples: {x_val_orig.shape[0]}')
    # print(f'Number of test samples: {x_test.shape[0]}')

    # --- prepare ground truth data in required CNN output format ---
    anchor_stride = 8  # NOTE: depends on the model configuration

    # iou thresholds for positive and negative samples
    iou_low = 0.5
    iou_high = 0.6
    num_negs_ratio = 5  # select X times more negative than positive samples

    # --- form output matrices ---
    # NOTE: images containing no objects, or objects which are not fully encased in an anchor, are discarded

    # generate masks of possible object locations
    obj_masks_train = helper_data.generate_anchor_level_object_masks(bboxes_train, img_dims, anchor_stride)
    obj_masks_val = helper_data.generate_anchor_level_object_masks(bboxes_val, img_dims, anchor_stride)

    # generate ground truth matrices
    result = Parallel(n_jobs=-1, temp_folder="/tmp", max_nbytes=None, backend="multiprocessing") \
        (delayed(helper_data.get_anchorless_ground_truth_data_parallel_optimized_limits)(bboxes_train[i],
                             obj_masks_train[:, :, i], img_dims, anchor_stride, iou_low, iou_high, num_negs_ratio)
         for i in tqdm(range(len(bboxes_train)), desc='Creating training ground truth anchor data...'))

    y_class_train, y_reg_train, valid_train = helper_data.parse_anchor_data_results(result)

    result = Parallel(n_jobs=-1, temp_folder="/tmp", max_nbytes=None, backend="multiprocessing") \
        (delayed(helper_data.get_anchorless_ground_truth_data_parallel_optimized_limits)(bboxes_val[i],
                             obj_masks_val[:, :, i], img_dims, anchor_stride, iou_low, iou_high, num_negs_ratio)
         for i in tqdm(range(len(bboxes_val)), desc='Creating validation ground truth anchor data...'))

    y_class_val, y_reg_val, valid_val = helper_data.parse_anchor_data_results(result)

    # normalize regression data
    reg_norm_coef = np.max(np.abs(y_reg_train))
    y_reg_train = y_reg_train / reg_norm_coef

    y_reg_val = y_reg_val / reg_norm_coef

    # save normalization coefficient
    f = open(os.path.join(modelsPath, 'norm_coef.txt'), 'w')
    f.write(str(reg_norm_coef))
    f.close()

    print(f'Ground truth of training set shape - classifier: {y_class_train.shape}')
    print(f'Ground truth of validation set shape - classifier: {y_class_val.shape}')

    print(f'Ground truth of training set shape - regressor: {y_reg_train.shape}')
    print(f'Ground truth of validation set shape - regressor: {y_reg_val.shape}')

    print(f'Number of positive samples in training set: {np.sum(y_class_train[:, :, :, 0])}')
    print(f'Number of positive samples in validation set: {np.sum(y_class_val[:, :, :, 0])}')


    # --- remove images containing no objects ---
    x_train = []
    for valid_ind in valid_train:
        x_train.append(x_train_orig[valid_ind])
    x_train = np.array(x_train)

    x_val = []
    for valid_ind in valid_val:
        x_val.append(x_val_orig[valid_ind])
    x_val = np.array(x_val)

    # # plot ground truth data
    plot_color = (255, 255, 255)
    prob_thr = 0.5

    helper_data.save_results_anchorless_limits_cls(gtDstPath, x_val, plot_color, y_class_val, y_reg_val, anchor_stride,
                                                   prob_thr, reg_norm_coef)

    # exit(1)

    # --- construct model ---
    # optimization hyperprameters
    epochs = 60
    lr = 0.0001
    batch_size = 10  # number of samples to process before updating the weights

    model = helper_model.construct_model_ssd_anchorless(input_shape=img_dims)  # build model architecture

    # compile model
    model.compile(loss={
        'rpn_out_class': helper_losses.rpn_loss_cls_multilabel,   # loss function applied to the layer named rpn_out_class
        'rpn_out_reg': helper_losses.rpn_loss_reg,
    },
        optimizer=Adam(lr=lr),
        metrics=['accuracy'])

    # --- fit model ---
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(modelsPath, 'checkpoint-{epoch:03d}-{val_loss:.4f}.hdf5'),
        # epoch number and val accuracy will be part of the weight file name
        monitor='val_loss',  # metric to monitor when selecting weight checkpoints to save
        verbose=1,
        save_best_only=False)  # True saves only the weights after epochs where the monitored value (val accuracy) is improved

    history = model.fit(x_train, y={"rpn_out_class": y_class_train, "rpn_out_reg": y_reg_train},
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[model_checkpoint],
                        verbose=1,
                        validation_data=(x_val, {"rpn_out_class": y_class_val, "rpn_out_reg": y_reg_val}),
                        shuffle=True)


    # --- save model ---
    # save model architecture
    print(model.summary())  # parameter info for each layer
    with open(os.path.join(modelsPath, 'modelSummary.txt'), 'w') as fh:  # save model summary
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    plot_model(model, to_file=os.path.join(modelsPath, 'modelDiagram.png'),
               show_shapes=True)  # save diagram of model architecture

    # save model configuration and weights
    model_json = model.to_json()  # serialize model architecture to JSON
    with open(os.path.join(os.path.join(modelsPath, 'model.json')), "w") as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(modelsPath, 'model.h5'))  # serialize weights to HDF5
    print("Saved model to disk.")

    # --- save training curves and logs ---
    helper_stats.save_training_logs_ssd(history=history, dst_path=modelsPath)

    # --- apply model to test data ---
    # [output_cls, output_reg] = model.predict(x_test, verbose=1)
