import os
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Input, concatenate,Cropping2D
from keras.models import Model, model_from_json
from tensorflow.keras.utils import plot_model


def load_model(model_path, weights_path):
    """
    loads a pre-trained model configuration and calculated weights
    :param model_path: path of the serialized model configuration file (.json) [string]
    :param weights_path: path of the serialized model weights file (.h5) [string]
    :return: model - keras model object
    """

    # --- load model configuration ---
    json_file = open(model_path, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)     # load model architecture

    model.load_weights(weights_path)     # load weights

    return model


def construct_model_anchorless_detector(input_shape):
    """
    construct region proposal model architecture
    classifier architecture for binary classification
    :param input_shape: list of input dimensions (height, width, depth) [tuple]
    :param num_anchors: number of different anchors with the same center [int]
    :return:  model - Keras model object
    """

    input_layer = Input(shape=input_shape)

    # feature extractor
    f1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(
        input_layer)
    f2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f1)
    f3 = MaxPool2D(pool_size=(2, 2))(f2)

    f4 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f3)
    f5 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f4)

    f6 = MaxPool2D(pool_size=(2, 2))(f5)

    f7 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f6)
    f8 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f7)
    f9 = MaxPool2D(pool_size=(2, 2))(f8)
    #
    f10 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f9)
    f11 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f10)
    # f11_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(
    #     f11)
    # # f12_0 = MaxPool2D(pool_size=(2, 2))(f11_1)
    # f12 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(
    #     f11_1)
    # f14 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f12)
    # classifier
    x_class_1 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu',
                       kernel_initializer='normal')(f11)
    x_class_2 = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='sigmoid',  # + 1 for the background class
                       kernel_initializer='uniform', name="out_class")(x_class_1)

    # regressor
    x_reg_1 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f11)
    # x_reg_1_0 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='normal')(x_reg_1)

    x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear',
                     kernel_initializer='zeros', name="out_reg")(x_reg_1)

    model = Model(input_layer, [x_class_2, x_reg_2])

    return model

def construct_model_anchorless_detector_inceptions(input_shape):
    input_layer = Input(shape=input_shape)

    # feature extractor
    f1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(
        input_layer)
    f2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f1)
    f3 = MaxPool2D(pool_size=(2, 2))(f2)

    f4 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f3)
    f5 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f4)

    f6 = MaxPool2D(pool_size=(2, 2))(f5)

    f7 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f6)
    f8 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f7)
    f9 = MaxPool2D(pool_size=(2, 2))(f8)
    # inception 1
    f10 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f9)
    f10_1 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f9)
    f10_con = concatenate([f10_1, f10], axis=3)

    f11 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)

    # f11_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(
    #     f11)
    # # f12_0 = MaxPool2D(pool_size=(2, 2))(f11_1)
    # f12 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(
    #     f11_1)
    # f14 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f12)

    f11_11 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)

    f_55 = Conv2D(filters=48, kernel_size=(5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)

    f_con = concatenate([f11_11, f11, f_55], axis=3)

    # version2
    # f11_11= Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f10)
    # f_55=Conv2D(filters=48, kernel_size=(5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(f11_11)

    # classifier
    x_class_1 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f_con)
    x_class_2 = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='sigmoid',  # + 1 for the background class
                       kernel_initializer='uniform', name="out_class")(x_class_1)

    # regressor
    x_reg_1 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f_con)
    # x_reg_1_0 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='normal')(x_reg_1)

    x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear',
                     kernel_initializer='zeros', name="out_reg")(x_reg_1)

    model = Model(input_layer, [x_class_2, x_reg_2])
    return model

def construct_model_anchorless_detector_inception(input_shape):
    """
    construct region proposal model architecture
    classifier architecture for binary classification
    :param input_shape: list of input dimensions (height, width, depth) [tuple]
    :param num_anchors: number of different anchors with the same center [int]
    :return:  model - Keras model object
    """

    input_layer = Input(shape=input_shape)

    # feature extractor
    f1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(
        input_layer)
    f2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f1)
    f3 = MaxPool2D(pool_size=(2, 2))(f2)

    f4 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f3)
    f5 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f4)

    f6 = MaxPool2D(pool_size=(2, 2))(f5)

    f7 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f6)
    f8 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f7)
    f9 = MaxPool2D(pool_size=(2, 2))(f8)
    #inception 1
    f10 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f9)
    f10_1 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f9)
    f10_con=concatenate([f10_1, f10], axis=3)



    f11 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)


    # f11_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(
    #     f11)
    # # f12_0 = MaxPool2D(pool_size=(2, 2))(f11_1)
    # f12 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(
    #     f11_1)
    # f14 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f12)


    f11_11 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)

    f_55=Conv2D(filters=48, kernel_size=(5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)

    f_con=concatenate([f11_11, f11, f_55], axis=3)

    #version2
    # f11_11= Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f10)
    # f_55=Conv2D(filters=48, kernel_size=(5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(f11_11)






    # classifier
    x_class_1 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu',
                       kernel_initializer='normal')(f_con)
    x_class_2 = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='sigmoid',  # + 1 for the background class
                       kernel_initializer='uniform', name="out_class")(x_class_1)

    # regressor
    x_reg_1 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f_con)
    # x_reg_1_0 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='normal')(x_reg_1)

    x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear',
                     kernel_initializer='zeros', name="out_reg")(x_reg_1)

    model = Model(input_layer, [x_class_2, x_reg_2])

    return model


def construct_model_anchorless_detector_skip_v1(input_shape):
    """
    construct region proposal model architecture
    classifier architecture for binary classification
    :param input_shape: list of input dimensions (height, width, depth) [tuple]
    :param num_anchors: number of different anchors with the same center [int]
    :return:  model - Keras model object
    """
#&NOTE receptive field na vakva mreza: 46px
    input_layer = Input(shape=input_shape)

    # feature extractor
    f1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(
        input_layer)
    f2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f1)
    f3 = MaxPool2D(pool_size=(2, 2))(f2)

    f4 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f3)
    f5 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f4)
    f6 = MaxPool2D(pool_size=(2, 2))(f5)

    f7 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f6)
    f8 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f7)
    f9 = MaxPool2D(pool_size=(2, 2))(f8)

    skip1 = Conv2D(filters=48, kernel_size=(5, 5), strides=(4, 4), padding='same', activation='relu', kernel_initializer='normal')(f3)
    skip1= Cropping2D(cropping=((1, 0), (0, 0)))(skip1)
    concat_skip = concatenate([skip1, f9])

    # inception 1:
    f10 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(concat_skip)
    f10_1 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(concat_skip)
    f10_con = concatenate([f10_1, f10], axis=3)

    # inception 2:
    f11 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)
    f11_11 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)
    f_55 = Conv2D(filters=48, kernel_size=(5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)
    f_con = concatenate([f11_11, f11, f_55], axis=3)

    # classifier
    x_class_1 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f_con)
    x_class_2 = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='sigmoid',  # + 1 for the background class
                       kernel_initializer='uniform', name="out_class")(x_class_1)

    # regressor
    x_reg_1 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f_con)
    x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear', kernel_initializer='zeros', name="out_reg")(x_reg_1)

    model = Model(input_layer, [x_class_2, x_reg_2])

    return model


if __name__ == '__main__':
    import helper_model1
    src_path = r'D:\Monika\Models\Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor1_All_Negatives_Renamed_alpha_1_reg_0.01_DIOU_LOSS_1_novi_fp_2_Focal_CATEGORICAL_BASE_normreg'

    model_path = os.path.join(src_path, 'model.json')
    weights_path = os.path.join(src_path, 'model.h5')
    new=os.path.join(src_path,'new')
    if not os.path.exists(new):
        os.makedirs(new, exist_ok=True)

    # im_size = (281, 469, 1)
    # im_size = (None, None, 1)
    # im_size = (540, 960, 1)
    im_size = (341, 512, 1)
    #im_size = (341, 512, 1)
    #im_size=(171,256,4)
    # model_single = helper_model1.construct_model_anchorless_detector_skip_v1(input_shape=im_size)
    model_single = helper_model1.construct_model_anchorless_detector_skip_v1_custom_loss_base(input_shape=im_size)
    # model_single = helper_model1.construct_model_anchorless_detector_Pedestrians_anchorless(input_shape=im_size)

    model_prev = load_model(model_path, weights_path)

    model_layer = model_prev.get_layer(name='model_1')
    weights_biases = model_layer.get_weights()

    cnt = 0
    for layer in model_prev.layers:
        print(layer.name, layer.trainable)
    for i in range(len(model_single.layers)):
        # print(model_single.layers[i].name, model_single.layers[i].trainable)
        if len(model_single.layers[i].trainable_weights) > 0:
            model_single.layers[i].set_weights([weights_biases[cnt], weights_biases[cnt + 1]])
            cnt += 2

    # --- save model ---
    # save model architecture
    print(model_single.summary())  # parameter info for each layer
    with open(os.path.join(os.path.join(new, 'modelSummary.txt')), 'w+') as fh:  # save model summary
        model_single.summary(print_fn=lambda x: fh.write(x + '\n'))
    # plot_model(model_single, to_file=os.path.join(r'C:\Users\User\Desktop\FolderForSharing\frli', 'modelDiagram.png'),
    #            show_shapes=True)  # save diagram of model architecture

    # save model configuration and weights
    model_json = model_single.to_json()  # serialize model architecture to JSON
    with open(os.path.join(os.path.join(new, 'model.json')), "w+") as json_file:
        json_file.write(model_json)
    model_single.save_weights(os.path.join(new, 'model.h5'))  # serialize weights to HDF5
    print("Saved model to disk.")
