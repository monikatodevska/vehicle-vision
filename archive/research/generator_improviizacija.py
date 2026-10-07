import os
import numpy as np
import pickle
# from keras.preprocessing.image import load_img, img_to_array
# from keras.utils import to_categorical
import cv2
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.utils.multi_gpu_utils import multi_gpu_model
from keras import backend as K
import helper_losses,helper_stats,helper_model1

def image_text_generator(data_dir,data_dir_gt, batch_size, img_height, img_width,reg_norm_coef_rows, reg_norm_coef_cols, reg_norm_coef_h,reg_norm_coef_w):
    """Generator that reads images and text from a directory and feeds them to a model.

    Args:
        data_dir (str): Path to directory containing images and pickled text files.
        batch_size (int): Number of samples per batch.
        img_height (int): Height of input image to the model.
        img_width (int): Width of input image to the model.

    Returns:
        A generator that yields batches of input images and corresponding text.
    """
    img_paths = sorted([os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.bmp')])
    txt_paths = sorted([os.path.join(data_dir_gt, fname) for fname in os.listdir(data_dir_gt) if fname.endswith('.txt')])

    # Create a list of unique image indices at the beginning of each epoch
    while True:
        indices = np.arange(len(img_paths))
        np.random.shuffle(indices)

        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            batch_imgs = np.zeros((len(batch_indices), img_height, img_width, 1))
            # batch_txt1 = []
            # batch_txt2 = []
            out_class_list=[]
            out_reg_list=[]

            for i, index in enumerate(batch_indices):
                # Load image and resize to desired dimensions
                image = cv2.imread(img_paths[index], 0)
                image = image.reshape(image.shape[0], image.shape[1], 1)

                # img = load_img(img_paths[index], target_size=(img_height, img_width))
                # img = img_to_array(img)
                batch_imgs[i] = image

                filename = txt_paths[index]
                fid=open(filename, 'rb')
                out_class_dims = pickle.load(fid)
                out_class_back = pickle.load(fid)
                out_reg_dims = pickle.load(fid)
                out_reg_back = pickle.load(fid)
                fid.close()

                out_class = np.reshape(out_class_back, (out_class_dims[0], out_class_dims[1], out_class_dims[2]))
                out_class_list.append(out_class)
                out_reg = np.reshape(out_reg_back, (out_reg_dims[0], out_reg_dims[1], out_reg_dims[2]))
                out_reg_list.append(out_reg)



                # Load pickled text file and extract data and dimensions
                # with open(txt_paths[index], 'rb') as f:
                #     data, shape1, shape2 = pickle.load(f)
                #     data = np.array(data)
                #     batch_txt1.append(data[:shape1])
                #     batch_txt2.append(data[shape1:])

            out_class_list = np.array(out_class_list)
            out_reg_list = np.array(out_reg_list)

            out_reg_norm = out_reg_list
            out_reg_norm[:, :, :, 0] = out_reg_list[:, :, :, 0] / reg_norm_coef_rows
            out_reg_norm[:, :, :, 1] = out_reg_list[:, :, :, 1] / reg_norm_coef_cols

            out_reg_norm[:, :, :, 2] = out_reg_list[:, :, :, 2] / reg_norm_coef_h
            out_reg_norm[:, :, :, 3] = out_reg_list[:, :, :, 3] / reg_norm_coef_w
            yield (batch_imgs, {'out_class':out_class_list, 'out_reg':out_reg_norm})





if __name__ == '__main__':



    version = 'Trening_With_Translations-Up-Down-All_Cams_NEW_Renamed_generator_improvizacija'
    # version = 'vinf_proba_ne'

    # --- paths ---
    srcImagesPath_Disk = r'D:\Monika\Trening_With_Translations-Up-Down-All_Cams_NEW_Renamed\Images'  # ova e eden folder
    # srcImagesPath_Ram = r'M:\Images'  # ova e eden folder
    GT_train_Disk = r'D:\Monika\Trening_With_Translations-Up-Down-All_Cams_NEW_Renamed\GT'
    # GT_train_Ram = r'M:\GT'

    srcImagesVal_Disk = r'D:\Monika\Trening_With_Translations-Up-Down-All_Cams_NEW_Renamed\Validacija\Images'
    GT_val_Disk = r'D:\Monika\Trening_With_Translations-Up-Down-All_Cams_NEW_Renamed\Validacija\GT'
    # srcImagesVal_Ram = srcImagesVal_Disk
    # GT_val_Ram = GT_val_Disk

    dstResultsPath = r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Results'
    dstModelsPath = r'D:\Monika\Models'
    dst_path_class_train = os.path.join(dstModelsPath, version, 'train_class.txt')
    dst_path_class_val = os.path.join(dstModelsPath, version, 'val_class.txt')
    dst_path_reg_train = os.path.join(dstModelsPath, version, 'train_reg.txt')
    dst_path_reg_val = os.path.join(dstModelsPath, version, 'train_val.txt')


    if not os.path.exists(os.path.join(dstModelsPath, version)):
        os.mkdir(os.path.join(dstModelsPath, version))
    modelsPath = os.path.join(dstModelsPath, version)
    if not os.path.exists(os.path.join(modelsPath, 'new')):
        os.mkdir(os.path.join(modelsPath, 'new'))


    file_path_reg_coef=r'D:\Monika\Trening_With_Translations-Up-Down-All_Cams_NEW_Renamed\reg_coef'

    OriginalModelsPath=r''

    # NOTE flags
    finetune = False

    # --- variables ---
    imgDims = {'rows': 341, 'cols': 512}
    # num_classes = 2
    img_depth = 1

    img_dims = (imgDims['rows'], imgDims['cols'], img_depth)


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
    # print(1)


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

    # --- fit model ---
    model_checkpoint = ModelCheckpoint(filepath=os.path.join(modelsPath, 'checkpoint-{epoch:03d}-{loss:.4f}.hdf5'),
                                       # epoch number and val accuracy will be part of the weight file name
                                       monitor='loss',  # metric to monitor when selecting weight checkpoints to save
                                       verbose=1,
                                       save_best_only=True)

    plot_losses = helper_stats.TrainingPlot(dst_path_class_train, dst_path_class_val,dst_path_reg_train, dst_path_reg_val)  # enable live plot of training and validation loss that updates after every epoch, optional



    train_idx_p = os.listdir(GT_train_Disk)
    val_idx_p = os.listdir(GT_val_Disk)

    train_generator = image_text_generator(srcImagesPath_Disk,GT_train_Disk, batch_size, img_dims[0], img_dims[1],reg_norm_coef_position_rows,reg_norm_coef_position_cols,reg_norm_coef_size_height,reg_norm_coef_size_width)
    val_generator = image_text_generator(srcImagesVal_Disk,GT_val_Disk, batch_size, img_dims[0], img_dims[1],reg_norm_coef_position_rows,reg_norm_coef_position_cols,reg_norm_coef_size_height,reg_norm_coef_size_width)

    # Train the model using the generator
    history=model.fit(train_generator,
                      steps_per_epoch=len(train_idx_p) // batch_size,
                      callbacks=[model_checkpoint, plot_losses],
                      verbose=1,
                      epochs=epochs,
                      validation_data=val_generator,
                      validation_steps=len(val_idx_p)//batch_size)

    print('model fitted')

    # history=model.fit(training_generator, epochs=epochs, steps_per_epoch=len(train_idx)/batch_size,
    #                   callbacks=[model_checkpoint, plot_losses],
    #                    verbose=1,
    #                   validation_data=validation_generator
    #                   )

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


