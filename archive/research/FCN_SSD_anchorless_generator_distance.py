
#import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
import os
# from tensorflow.keras.utils import plot_model
# from keras.callbacks import ModelCheckpoint
import random
#from tqdm import tqdm
from copy import deepcopy
from joblib import Parallel, delayed
import multiprocessing
import xml.etree.ElementTree as ET
import keras
# custom imports
import sys
sys.path.insert(0, 'E:\Science\Monika\SingleShotDetector\Helpers' )
import helper_postprocessing
import pickle
import helper_losses,helper_model,helper_postprocessing,helper_stats
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import helper_model1
from keras.utils.multi_gpu_utils import multi_gpu_model
from keras import backend as K


import helper_anchorless_mc
from joblib import Parallel, delayed
import multiprocessing
class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, list_IDs, image_path_ram, image_path_disk,  groundtruthfiles_ram,groundtruthfiles_disk,  anchor_stride, reg_norm_coef_rows,reg_norm_coef_cols, reg_norm_coef_h, reg_norm_coef_w, finetune,batch_size, shuffle, n_classes, to_fit=True):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
        self.image_path_ram = image_path_ram
        self.image_path_disk = image_path_disk
        self.groundtruthfiles_ram=groundtruthfiles_ram
        self.groundtruthfiles_disk=groundtruthfiles_disk
        self.anchor_stride = anchor_stride
        # self.im_size = im_size

        self.reg_norm_coef_rows = reg_norm_coef_rows
        self.reg_norm_coef_cols = reg_norm_coef_cols
        self.reg_norm_coef_h = reg_norm_coef_h
        self.reg_norm_coef_w = reg_norm_coef_w
        self.finetune=finetune

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_classes = n_classes

        self.to_fit = to_fit


        self.on_epoch_end()



    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index, out_reg_norm=None):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        valid_train=[]
        out_class=[]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        # X = self._generate_X(list_IDs_temp)
        # num_cores = multiprocessing.cpu_count()

        x_train_orig= self.read_data_rpn(list_IDs_temp, self.image_path_ram, self.image_path_disk, 1)
        # print(x_train_orig.shape)
        out_class, out_reg=self.get_output(list_IDs_temp, self.groundtruthfiles_ram, self.groundtruthfiles_disk)

        # out_reg_norm=out_reg/self.reg_norm_coef
        out_reg_norm=out_reg
        out_reg_norm[:, :, :, 0] = out_reg[:, :, :, 0] / self.reg_norm_coef_rows
        out_reg_norm[:, :, :, 1] = out_reg[:, :, :, 1] / self.reg_norm_coef_cols

        out_reg_norm[:, :, :, 2] = out_reg[:, :, :, 2] / self.reg_norm_coef_h
        out_reg_norm[:, :, :, 3] = out_reg[:, :, :, 3] / self.reg_norm_coef_w
        # print(out_reg.shape)
        # print(1)
        return (x_train_orig, {'out_class':out_class, 'out_reg':out_reg_norm})

        # result = Parallel(n_jobs=num_cores, temp_folder="/tmp", max_nbytes=None, backend="multiprocessing")(delayed(self.get_anchor_data_ssd)(bbox_train, self.anchor_dims, self.im_size, self.anchor_stride, self.iou_low, self.iou_high) for bbox_train in bboxes_train)
        # for res_ind, results in enumerate(result):
        #     # print(results[res_ind])
        #     if result[res_ind] is None:
        #         print('k')
        #         continue
        #     valid_train.append(res_ind)
        #     out_class.append(results)
        # out_class_train = np.array(out_class)
        # x_train = []
        # for valid_ind in valid_train:
        #     x_train.append(x_train_orig[valid_ind])
        # x_train = np.array(x_train)
        # return x_train,out_class_train

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def read_data_rpn(self, list_id, im_path_ram, im_path_disk, im_depth):
        """
        load, resize, and normalize image data
        loads images and annotations from one folder
        :param im_path: global path of folder containing images of a data subset [string]
        :param im_size: output dimensions of the images (cols, rows) [tuple]
        :param im_depth: required depth of the loaded images (value: 1 or 3) [int]
        :param shuffle: whether to shuffle input data order [bool]
        :return: images_list - array of normalized depth maps [ndarray]
        object_annotations_list - annotated bounding boxes min_row, min_col, max_row, max_col [list] """
        #im_path_resized=r'E:\Science\Monika\M-30HD_resized'
        images_list = []       # array of normalized images
        object_annotations_list = []       # array of array of bounding boxes for each image
        # WScale=512/1200
        # HScale=480/720
        objects=[]
        # list images in source folder
        for i, ID in enumerate(list_id):
            # Store sample
            # flag1[i], X[i,] = self._load_grayscale_image(os.path.join(image_path, 'image' + str(ID).zfill(6) + '.jpg'), self.img_dims)
            # images_list1.append(X[i,])
            im_name=str('image' + str(ID).zfill(6) + '.bmp')
            # print(im_name)
            # print(im_path)
            flag1 = 0
            #print(im_name)
            #cv2.waitKey(0)
            # --- load image ---
            # if not im_name[-4:] != '.bmp':  # exclude system files
            #     continue

            # if im_depth == 3:
            #     image = cv2.imread(os.path.join(im_path, im_name))
            # else:



            # image = cv2.imread(os.path.join(im_path_ram, im_name), 0)
            # if image is None:
            image = cv2.imread(os.path.join(im_path_disk, im_name), 0)

            # if image is None:
            #     print(os.path.join(im_path, im_name))
            #     exit(1)
            # rows, col=image.shape[:2]
            # if im_size != (col,rows):
            #     image = cv2.resize(image, im_size, interpolation=cv2.INTER_AREA)
            image = image.reshape(image.shape[0], image.shape[1], im_depth)



            images_list.append(image)

        if len(images_list) == 0:
            print("No images were read.")
            exit(100)



        images_list = np.array(images_list).astype(np.uint8)

        return images_list

    def get_output(self, list_ids, groundtruthfiles_ram,groundtruthfiles_disk):
        out_class_list=[]
        out_reg_list=[]
        for i, id in enumerate(list_ids):
            filename = 'image' +str(id).zfill(6) + '.txt'
            # if os.path.exists(os.path.join(groundtruthfiles_ram, filename)):
            #     fid1=open(os.path.join(groundtruthfiles_ram, filename), 'rb')
            # else:
            fid1=open(os.path.join(groundtruthfiles_disk, filename), 'rb')

            out_class_dims=pickle.load(fid1)
            out_class_back=pickle.load(fid1)
            out_reg_dims=pickle.load(fid1)
            out_reg_back=pickle.load(fid1)
            fid1.close()
            out_class= np.reshape(out_class_back, (out_class_dims[0], out_class_dims[1], out_class_dims[2]))
            out_class_list.append(out_class)
            out_reg = np.reshape(out_reg_back, (out_reg_dims[0], out_reg_dims[1], out_reg_dims[2]))
            out_reg_list.append(out_reg)
        out_class_list=np.array(out_class_list)
        out_reg_list=np.array(out_reg_list)
        # print(out_class_list)
        return out_class_list, out_reg_list



if __name__ == '__main__':



    # version = 'Trening_With_Translations-Up-Down-All_Cams_Wavelet_NAJNOVI_Renamed_ORIGINAL_BEZ_WAVELET_4_FILTERS_FIRST_VGG_V3'
    version = 'Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor_DISTANCE_Renamed_pravilno'
    # version = 'vinf_proba_ne'_pravilno

    # --- paths ---
    srcImagesPath_Disk = r'M:\Monika\Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor_DISTANCE_Renamed\Images'  # ova e eden folder
    srcImagesPath_Ram = r'M:\Images'  # ova e eden folder
    GT_train_Disk = r'M:\Monika\Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor_DISTANCE_Renamed\GT'
    GT_train_Ram = r'M:\GT'

    srcImagesVal_Disk = r'M:\Monika\Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor_DISTANCE_Renamed\Validacija\Images'
    GT_val_Disk = r'M:\Monika\Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor_DISTANCE_Renamed\Validacija\GT'
    srcImagesVal_Ram = srcImagesVal_Disk
    GT_val_Ram = GT_val_Disk

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

    # if not os.path.exists(os.path.join(dstModelsPath, version, 'reg_coef')):
    #     os.mkdir(os.path.join(dstModelsPath, version, 'reg_coef'))
    # file_path_reg_coef = os.path.join(dstModelsPath, version, 'reg_coef')

    file_path_reg_coef=r'M:\Monika\Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor_DISTANCE_Renamed\reg_coef'

    OriginalModelsPath=r''

    # NOTE flags
    finetune = False

    # --- variables ---
    imgDims = {'rows': 341, 'cols': 512}
    # num_classes = 2
    img_depth = 1

    img_dims = (imgDims['rows'], imgDims['cols'], img_depth)

    train_idx_p = os.listdir(GT_train_Disk)
    train_idx = []
    for i, im_name in enumerate(train_idx_p):
        index = int(im_name[5:11])
        train_idx.append(index)

    val_idx_p = os.listdir(GT_val_Disk)
    val_idx = []
    for i, im_name in enumerate(val_idx_p):
        index = int(im_name[5:11])
        val_idx.append(index)

    anchor_stride = 8

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
    print(1)


    training_generator = DataGenerator(train_idx, srcImagesPath_Ram, srcImagesPath_Disk, GT_train_Ram, GT_train_Disk, anchor_stride,reg_norm_coef_position_rows,reg_norm_coef_position_cols,reg_norm_coef_size_height,reg_norm_coef_size_width, finetune,
                                                  batch_size=64,  shuffle=True,  n_classes=4,to_fit=True)
    # k=training_generator.__getitem__(0)
    # # print(len(k))
    # print([a.shape for a in k])
    # print(k.shape)
    # np.max(k)

    validation_generator = DataGenerator(val_idx, srcImagesVal_Ram, srcImagesVal_Disk, GT_val_Ram, GT_val_Disk, anchor_stride, reg_norm_coef_position_rows,reg_norm_coef_position_cols,reg_norm_coef_size_height,reg_norm_coef_size_width, finetune,
                                                   batch_size=64,  shuffle=True,  n_classes=4,to_fit=True)

    epochs = 30
    lr = 0.0001
    batch_size = 64
    im_size = (341, 512, 1)
    plot_color = (0, 0, 255)
    prob_thr = 0.5

    # anchorless_genertor_plot.save_results_anchorless_limits_cls(resultspath, x_train, out_class_train, out_reg_train, anchor_stride, prob_thr)
    if finetune:
        model = helper_model1.load_model(model_path=os.path.join(OriginalModelsPath, 'model.json'),
                                         weights_path=os.path.join(OriginalModelsPath, 'model.h5'))
    else:
        # model = helper_model1.construct_model_anchorless_detector_wavelet_simple(input_shape=im_size)  # build model architecture
        model = helper_model1.construct_model_anchorless_detector_skip_v1(input_shape=im_size)  # build model architecture
    # model = helper_model1.load_model(r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Models\vinf_1000_SO_DVA_INCEPTION_SKIP_48_normalizationWithMax_0.45_0.7\novi_bagnat\model.json', r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Models\vinf_1000_SO_DVA_INCEPTION_SKIP_48_normalizationWithMax_0.45_0.7\novi_bagnat\model.h5')

    # model_layer = model.get_layer()
    # weights_biases = model.get_weights()
    fixed_filters = False
    if fixed_filters:
        filter = helper_model1.my_filter(shape=(3, 3, 1, 8), dtype=None)
        biases = np.zeros((8,))
        model.layers[1].set_weights([filter, biases])

        model.layers[1].trainable = False
        filter_2 = helper_model1.my_filter_2(shape=(3, 3, 32, 8), dtype=None)
        model.layers[4].set_weights([filter_2, biases])
        model.layers[4].trainable = False



    # weights_biases1 = model.get_weights()
    # print(weights_biases1[0][:,:,0,0])
    # for ind, layer in enumerate(model.layers):
    #
    #     print(layer.name)
    #     print("Weights")
    #     print("Shape: ", layer.get_weights()[0])
    #     print("Bias")
    #     print("Shape: ",  layer.get_weights()[1], '\n')

    # cnt = 0
    # for i in range(len(model_single.layers)):
    #     if len(model_single.layers[i].trainable_weights) > 0:
    #         model_single.layers[i].set_weights([weights_biases[cnt], weights_biases[cnt + 1]])
    #         cnt += 2
    # biases=np.zeros((8,))
    # f1_p1.trainable = False

    # f1_p1.set_weights([filter, biases])
    # weights=f1_p1.get_weights()

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
    # import focal_loss
    # compile model

    model.compile(loss={
        'out_class': helper_losses.rpn_loss_cls_new,
        'out_reg': helper_losses.rpn_loss_reg

    },
        optimizer=Adam(lr=lr),
        metrics=['accuracy'])

    # # compile model
    # model.compile(loss={
    #     'out_class': helper_losses.rpn_loss_cls_new,
    #     'out_reg': helper_losses.rpn_loss_reg
    #
    # },
    #     optimizer=Adam(lr=lr),
    #     metrics=['accuracy'])

    # --- fit model ---
    model_checkpoint = ModelCheckpoint(filepath=os.path.join(modelsPath, 'checkpoint-{epoch:03d}-{loss:.4f}.hdf5'),
                                       # epoch number and val accuracy will be part of the weight file name
                                       monitor='loss',  # metric to monitor when selecting weight checkpoints to save
                                       verbose=1,
                                       save_best_only=True)

    plot_losses = helper_stats.TrainingPlot(dst_path_class_train, dst_path_class_val,dst_path_reg_train, dst_path_reg_val)  # enable live plot of training and validation loss that updates after every epoch, optional

    # True saves only the weights after epochs where the monitored value (val accuracy) is improved
    # # history = model.fit_generator(training_generator,
    #                               epochs=epochs,
    #                               # steps_per_epoch=len(train_idx)/batch_size,
    #                               callbacks=[model_checkpoint, plot_losses],
    #                               verbose=1,
    #                               validation_data=validation_generator
    #                               )

    print('model fitted')

    history=model.fit(training_generator, epochs=epochs, steps_per_epoch=len(train_idx)/batch_size,
                      callbacks=[model_checkpoint, plot_losses],
                      verbose=1,
                      use_multiprocessing=True,
                      max_queue_size=70,
                      workers=10,
                      validation_data=validation_generator
                      )

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
    model = helper_model1.repack_as_single_gpu(model, func_construct_model, im_size, i=1,finetune=False)
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

