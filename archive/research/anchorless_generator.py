
#import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
import os
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
import helper_anchorless_mc
from joblib import Parallel, delayed
import multiprocessing
class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, list_IDs, image_path,  groundtruthfiles,  anchor_stride, anchor_dims, iou_low, iou_high, im_size,  img_depth, reg_norm_coef, batch_size, exclude_empty, shuffle,  to_fit=True, n_classes=1,):
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
        self.image_path = image_path
        # self.annot_path=annot_path_train
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.im_size = im_size
        # self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.anchor_stride=anchor_stride
        self.anchor_dims=anchor_dims
        self.iou_low=iou_low
        self.iou_high=iou_high
        self.exclude_empty=exclude_empty
        self.img_depth=img_depth
        self.groundtruthfiles=groundtruthfiles
        self.reg_norm_coef=reg_norm_coef
    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
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

        x_train_orig= self.read_data_rpn(list_IDs_temp, self.image_path, (self.im_size[1], self.im_size[0]), self.img_depth,  exclude_empty=True)
        # print(x_train_orig.shape)
        out_class, out_reg=self.get_output(list_IDs_temp, self.groundtruthfiles)
        out_reg_norm=out_reg/self.reg_norm_coef
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


    def read_data_rpn(self, list_id, im_path, im_size, im_depth, exclude_empty):
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
            im_name=str('image' + str(ID).zfill(6) + '.jpg')
            # print(im_name)
            flag1 = 0
            #print(im_name)
            #cv2.waitKey(0)
            # --- load image ---
            # if not im_name[-4:] != '.bmp':  # exclude system files
            #     continue

            # if im_depth == 3:
            #     image = cv2.imread(os.path.join(im_path, im_name))
            # else:
            image = cv2.imread(os.path.join(im_path, im_name), 0)
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

    def get_output(self, list_ids, groundtruthfiles):
        out_class_list=[]
        out_reg_list=[]
        for i, id in enumerate(list_ids):
            filename = 'image' +str(id).zfill(6) + '.txt'
            fid1=open(os.path.join(groundtruthfiles, filename), 'rb')
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
    version = 'vinf_156_dotreniranje_2'

    groundtruthfilesTrain=r'D:\KlasifikacijaVozila\Miladinovci\GroundTruthTrain'
    groundtruthfilesVal=r'D:\KlasifikacijaVozila\Miladinovci\GroundTruthVal'

    srcImagesPath = r'D:\KlasifikacijaVozila\Miladinovci\Images_Train'
    # srcImagesPathTrain = r'E:\Science\Monika1\train1'
    srcImagesPathVal = r'D:\KlasifikacijaVozila\Miladinovci\Images_Val'
    # srcAnnotationsPath = r'E:\Science\Monika\GRAM-RTMv4\Annotations\site2'
    # srcAnnotationsPathTrain = r'E:\Science\Monika\GRAM-RTMv4\Annotations\M-30-anotacii'
    dstResultsPath = r'E:\Science\Monika\Results'
    dstModelsPath = r'E:\Science\Monika\Models'
    gtDstPath = r'E:\Science\Monika\GT'
    file_path_reg_coef = r'D:\KlasifikacijaVozila\Miladinovci'

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
    imgDims = {'rows': 341, 'cols':512 }

    num_classes = 1
    img_depth = 1
    im_size=[341, 512,1]

    anchor_dims = ((32,32), (48,48), (64,64), (92,92))
    anchor_stride = 8
    norm_coef = 100  # constant to normalize regression ground truth data

    # IOU thresholds for selecting positive and negative anchors
    iou_low = 0.4
    iou_high = 0.6
    train_idx_p = os.listdir(groundtruthfilesTrain)
    train_idx = []
    for i, im_name in enumerate(train_idx_p):
        index = int(im_name[5:11])
        train_idx.append(index)

    val_idx_p = os.listdir(groundtruthfilesVal)
    val_idx = []
    for i, im_name in enumerate(val_idx_p):
        index = int(im_name[5:11])
        val_idx.append(index)

    # print(1)
    filename='reg_norm_coef.txt'
    fid1 = open(os.path.join(file_path_reg_coef, filename), 'rb')
    reg_norm_coef=pickle.load(fid1)
    print(reg_norm_coef)
    training_generator = DataGenerator(train_idx, srcImagesPath,groundtruthfilesTrain, anchor_stride, anchor_dims, iou_low, iou_high, im_size, img_depth,reg_norm_coef,
                                                  batch_size=40, exclude_empty=False, shuffle=True, to_fit=True, n_classes=1)
    # k=training_generator.__getitem__(0)
    # # print(len(k))
    # print([a.shape for a in k])
    # print(k.shape)
    # np.max(k)

    validation_generator = DataGenerator(val_idx, srcImagesPathVal,groundtruthfilesVal, anchor_stride, anchor_dims, iou_low, iou_high, im_size, img_depth, reg_norm_coef, batch_size=40,
                                                   exclude_empty=False, shuffle=False, to_fit=True, n_classes=1)

    epochs = 5
    lr = 0.00005
    batch_size = 32

    # model = helper_model1.construct_model_anchorless_detector(input_shape=im_size)  # build model architecture
    model=helper_model1.load_model(r'E:\Science\Monika\Models\vinf_156_dotreniranje\model.json', r'E:\Science\Monika\Models\vinf_156_dotreniranje\model.h5')
    # compile model
    model.compile(loss={
        'out_class': helper_losses.rpn_loss_cls,
        'out_reg': helper_losses.rpn_loss_reg

    },
        optimizer=Adam(lr=lr),
        metrics=['accuracy'])

    # --- fit model ---
    model_checkpoint = ModelCheckpoint(filepath=os.path.join(modelsPath, 'checkpoint-{epoch:03d}-{loss:.4f}.hdf5'),  # epoch number and val accuracy will be part of the weight file name
                                       monitor='loss',  # metric to monitor when selecting weight checkpoints to save
                                       verbose=1,
                                       save_best_only=True)  # True saves only the weights after epochs where the monitored value (val accuracy) is improved

    history = model.fit_generator(training_generator,
                                  epochs=epochs,
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