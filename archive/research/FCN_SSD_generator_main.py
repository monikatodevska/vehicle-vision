
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
import FCN_SSD_gen

import h5py
# custom package imports
#from SingleShotDetector.Helpers import helper_model, helper_data, helper_stats, losses
import sys
#sys.path.insert(0, 'E:\Science\Monika\SingleShotDetector\Helpers' )
#import cv2

import helper_model, helper_stats, helper_losses
import helper_data

# --- paths ---
version = 'debug_200'

srcImagesPathTrain = r'E:\Science\Monika\train-proba'
srcAnnotationsPath = r'E:\Science\Monika\GRAM-RTMv4\Annotations\site2'
srcImagesPathVal = r'E:\Science\Monika\val1'
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

img_dims = (imgDims['rows'], imgDims['cols'])

train_idx_p=os.listdir(srcImagesPathTrain)
train_idx=[]
for i, im_name in enumerate(train_idx_p):
    index=int(im_name[5:11])
    train_idx.append(index)

val_idx_p=os.listdir(srcImagesPathVal)
val_idx=[]
for i, im_name in enumerate(val_idx_p):
    index=int(im_name[5:11])
    val_idx.append(index)

print(1)
# --- load and format data ---

# load full dataset into memory - image data and labels
#x_train_orig, bboxes_train = helper_data.read_data_rpn(srcImagesPath, (imgDims['cols'], imgDims['rows']), img_depth, srcAnnotationsPath, exclude_empty=True, shuffle=False)

#x_train_orig, bboxes_train = helper_data.read_data_rpn(os.path.join(srcImagesPath, 'train2'), (imgDims['cols'], imgDims['rows']), img_depth, srcAnnotationsPath, exclude_empty=True, shuffle=False)
#x_val_orig, bboxes_val = helper_data.read_data_rpn(os.path.join(srcImagesPath, 'val2'), (imgDims['cols'], imgDims['rows']), img_depth, srcAnnotationsPath, exclude_empty=True, shuffle=False)
# x_test, bboxes_test = helper_data.read_data_rpn(os.path.join(srcImagesPath, 'test'), (imgDims['cols'], imgDims['rows']), img_depth, srcAnnotationsPath, exclude_empty=False)

# print(f'Training dataset shape: {x_train_orig.shape}')
# print(f'Number of training samples: {x_train_orig.shape[0]}')
# print(f'Number of validation samples: {x_val_orig.shape[0]}')
# print(f'Number of test samples: {x_test.shape[0]}')
h=[]
w=[]
#bboxes_train=[[[1,2,3,4],[5,6,7,8]], [[9,8,7,0],[4,6,4,2]]]
# for box_ind, bbox in enumerate(bboxes_train):
#     for rect in bbox:
#         h=np.append(h,(rect[2]-rect[0]))
#         w=np.append(w,(rect[3]-rect[1]))
#
# h_l=h.tolist()
# w_l=w.tolist()
# s1=[2]
# plt.scatter(h_l,w_l,s=s1)
# plt.show()
#plot ground truth annotations ---
# gt_annotated_images_dst_path = r'E:\Science\Monika\anotirani_groundtruth'
# x_train_orig_2 = deepcopy(x_train_orig)
#
# helper_data.plot_gt_annotations(x_train_orig_2, bboxes_train, gt_annotated_images_dst_path)



# --- prepare ground truth data in required format ---

# anchor parameters
anchor_dims = ((25,25),(32, 32),(48,48), (64, 64), (92, 92))
anchor_stride = 8
norm_coef = 100     # constant to normalize regression ground truth data

# IOU thresholds for selecting positive and negative anchors
iou_low = 0.3
iou_high = 0.5

# generate ground truth output
#  print(len(bboxes_train));
# result = Parallel(n_jobs=-1, temp_folder="/tmp", max_nbytes=None, backend="multiprocessing")(delayed(helper_data.get_anchor_data_ssd)
#                                                                                              (bbox_train, anchor_dims, img_dims, anchor_stride, iou_low, iou_high) for bbox_train in bboxes_train)
# for i, el in enumerate(result):
#     valid_train.append(i)
#     y_class_train.append(el[0])
#     y_reg_train.append(el[1])
# y_class_train=np.array(y_class_train)
# y_reg_train=np.array(y_reg_train)

#y_class_train, y_reg_train, valid_train = helper_data.get_anchor_data_ssd(bboxes_train, anchor_dims, img_dims, anchor_stride, iou_low, iou_high)
#y_class_val, y_reg_val, valid_val = helper_data.get_anchor_data_ssd(bboxes_val, anchor_dims, img_dims, anchor_stride, iou_low, iou_high)
# y_class_test, valid_test = helper_data.get_anchor_data_cls(bboxes_test, anchor_dims, img_dims, anchor_stride, iou_low, iou_high)

# remove images without valid objects
# x_train = []
# for valid_ind in valid_train:
#     x_train.append(x_train_orig[valid_ind])
# x_train = np.array(x_train)
# print(len(x_train))
# x_val = []
# for valid_ind in valid_val:
#     x_val.append(x_val_orig[valid_ind])
# x_val = np.array(x_val)
#
# print(len(y_class_train))
#
# # --- plot ground truth network output ---
# x_val_2 = deepcopy(x_val)
# prob_thr = 0.5
#
# ground_truth_annotations_anchors_path_cls = r'C:\\Users\\User\\Desktop\\annotations_vis_gt'
# ground_truth_annotations_anchors_path_reg = r'C:\\Users\\User\\Desktop\\annotations_vis_gt_reg'
#
# output_branch = 'regressor'
#
 helper_data.save_results(ground_truth_annotations_anchors_path_reg, x_val_2, (255, 255, 255), y_class_val, y_reg_val, anchor_dims, anchor_stride, prob_thr, norm_coef, output_branch)



# --- construct model ---

# optimization hyperprameters
epochs = 30
lr = 0.0001
batch_size = 25

training_generator = FCN_SSD_gen.DataGenerator(train_idx,  srcImagesPathTrain, srcAnnotationsPath, anchor_stride, anchor_dims, iou_low, iou_high, img_dims, to_fit=True, batch_size=25,
                 n_channels=1, n_classes=1, shuffle=True)

validation_generator = FCN_SSD_gen.DataGenerator(val_idx,   srcImagesPathTrain, srcAnnotationsPath, anchor_stride, anchor_dims, iou_low, iou_high, img_dims, to_fit=True, batch_size=25,
                 n_channels=1, n_classes=1, shuffle=True)

print(1)
model = helper_model.construct_model_ssd_cls(input_shape=img_dims, num_anchors=len(anchor_dims))   # build model architecture

# compile model
model.compile(loss={
                  'out_class': helper_losses.rpn_loss_cls

                   },
              optimizer=Adam(lr=lr),
              metrics=['accuracy'])


# --- fit model ---
model_checkpoint = ModelCheckpoint(filepath=os.path.join(modelsPath, 'checkpoint-{epoch:03d}-{loss:.4f}.hdf5'),   # epoch number and val accuracy will be part of the weight file name
                                   monitor='loss',      # metric to monitor when selecting weight checkpoints to save
                                   verbose=1,
                                   save_best_only=True)     # True saves only the weights after epochs where the monitored value (val accuracy) is improved

history = model.fit_generator(training_generator,
                    batch_size=batch_size,  # number of samples to process before updating the weights
                    epochs=epochs,
                    callbacks=[model_checkpoint],
                    verbose=1,
                    validation_data=validation_generator)

print('model fitted')
# --- save model ---
# save model architecture
print(model.summary())      # parameter info for each layer
with open(os.path.join(modelsPath, 'modelSummary.txt'), 'w') as fh:     # save model summary
   model.summary(print_fn=lambda x: fh.write(x + '\n'))
#plot_model(model, to_file=os.path.join(modelsPath, 'modelDiagram.png'), show_shapes=True)   # save diagram of model architecture

# save model configuration and weights
model_json = model.to_json()  # serialize model architecture to JSON
with open(os.path.join(os.path.join(modelsPath, 'model.json')), "w") as json_file:
    json_file.write(model_json)
model.save_weights(os.path.join(modelsPath, 'model.h5'))  # serialize weights to HDF5

print('d')
# --- save training curves and logs ---
helper_stats.save_training_logs_ssd(history=history, dst_path=modelsPath)

# --- apply model to test data ---
# Y_test_pred = model.predict(x_test, verbose=1)
