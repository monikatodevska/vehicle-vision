
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, GlobalMaxPooling2D,
                                     Reshape, Add, Multiply, Activation, Lambda,
                                     Concatenate, Conv2D)
from tensorflow.keras.layers import *

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

    f10 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f9)
    # f10_1 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f9)
    # f10_con=concatenate([f10_1, f10], axis=3)



    f11 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f10)


    # f11_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(
    #     f11)
    # # f12_0 = MaxPool2D(pool_size=(2, 2))(f11_1)
    # f12 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(
    #     f11_1)
    # f14 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f12)

    #inception2
    # f11_11 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)
    #
    # f_55=Conv2D(filters=48, kernel_size=(5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)
    #
    # f_con=concatenate([f11_11, f11, f_55], axis=3)

    #version2
    # f11_11= Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f10)
    # f_55=Conv2D(filters=48, kernel_size=(5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(f11_11)


    # classifier
    x_class_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f11)
    x_class_2 = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='softmax',  # + 1 for the background class
                       kernel_initializer='uniform', name="out_class")(x_class_1)

    # regressor
    x_reg_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f11)
    # x_reg_1_0 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='normal')(x_reg_1)

    x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear',
                     kernel_initializer='zeros', name="out_reg")(x_reg_1)

    model = Model(input_layer, [x_class_2, x_reg_2])

    return model

def construct_model_anchorless_detector_Pedestrians_anchorless_64(input_shape):
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

    # f10 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f9)
    # f10_1 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f9)
    # f10_con=concatenate([f10_1, f10], axis=3)



    # f11 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f10)

    f10 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f9)
    f11= Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f10)

    # classifier
    x_class_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f11)
    x_class_2 = Conv2D(filters=2, kernel_size=(1, 1), padding='same', activation='sigmoid',  # + 1 for the background class
                       kernel_initializer='uniform', name="out_class")(x_class_1)

    # regressor
    # x_reg_1 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f9)
    # # x_reg_1_0 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='normal')(x_reg_1)
    #
    # x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear',
    #                  kernel_initializer='zeros', name="out_reg")(x_reg_1)

    # model = Model(input_layer, [x_class_2, x_reg_2])
    model = Model(input_layer, [x_class_2])

    return model


init1=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=3)
init2=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=5)
init3=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=7)
init4=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=10)
init5=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=15)
init6=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=20)
init7=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=4)
init8=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=8)
init9=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=30)

init_class=keras.initializers.RandomUniform(minval=0.0, maxval=1.0,seed=0.5)
import tensorflow
mirrored_strategy = tensorflow.distribute.MirroredStrategy(devices=["/job:localhost/replica:0/task:0/device:GPU:0", "/job:localhost/replica:0/task:0/device:GPU:1"],
                                                   cross_device_ops=tensorflow.distribute.HierarchicalCopyAllReduce())
print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))


with mirrored_strategy.scope():


    def construct_model_anchorless_detector_Pedestrians_anchorless(input_shape):
        """
        construct region proposal model architecture
        classifier architecture for binary classification
        :param input_shape: list of input dimensions (height, width, depth) [tuple]
        :param num_anchors: number of different anchors with the same center [int]
        :return:  model - Keras model object
        """
        #NOTE 56px receptive field so cekor 4(bez tretiot maxpooling

        #NOTE 84px receptive field so cekor 8

        input_layer = Input(shape=input_shape)


        # feature extractor
        f1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv1')(
            input_layer)
        f2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv2')(f1)
        f3 = MaxPool2D(pool_size=(2, 2))(f2)

        f4 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv3')(f3)
        f5 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv4')(f4)

        f6 = MaxPool2D(pool_size=(2, 2))(f5)

        f7 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv5')(f6)
        f8 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv6')(f7)
        f9 = MaxPool2D(pool_size=(2, 2))(f8)

        # f9=f8


        # f10 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f9)
        # f10_1 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f9)
        # f10_con=concatenate([f10_1, f10], axis=3)



        # f11 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f10)

        f10 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv7')(f9)
        f11= Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv8')(f10)

        x_class_1 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv9_class')(f11)

        # classifier

        x_class_2 = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid',  # + 1 for the background class
                           kernel_initializer='uniform', name="out_class")(x_class_1)

        x_weights = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid', kernel_initializer='uniform', name="weights_output")(x_class_1)

        combined_output = concatenate([x_class_2, x_weights], axis=-1)
        #regressor
        # x_reg_1 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f9)
        # # x_reg_1_0 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='normal')(x_reg_1)
        #
        # x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear',
        #                  kernel_initializer='zeros', name="out_reg")(x_reg_1)
        #
        # model = Model(input_layer, [x_class_2, x_reg_2])
        model = Model(input_layer, combined_output)

        return model



def construct_model_anchorless_detector_Pedestrians_anchorless_same_init(input_shape):
    """
    construct region proposal model architecture
    classifier architecture for binary classification
    :param input_shape: list of input dimensions (height, width, depth) [tuple]
    :param num_anchors: number of different anchors with the same center [int]
    :return:  model - Keras model object
    """
    #NOTE 56px receptive field so cekor 4(bez tretiot maxpooling

    #NOTE 84px receptive field so cekor 8

    input_layer = Input(shape=input_shape)


    # feature extractor
    f1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=init, name='conv1')(
        input_layer)
    f2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=init, name='conv2')(f1)
    f3 = MaxPool2D(pool_size=(2, 2))(f2)

    f4 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=init, name='conv3')(f3)
    f5 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=init, name='conv4')(f4)

    f6 = MaxPool2D(pool_size=(2, 2))(f5)

    f7 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=init, name='conv5')(f6)
    f8 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=init, name='conv6')(f7)
    f9 = MaxPool2D(pool_size=(2, 2))(f8)

    # f9=f8


    # f10 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f9)
    # f10_1 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f9)
    # f10_con=concatenate([f10_1, f10], axis=3)



    # f11 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f10)

    f10 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=init, name='conv7')(f9)
    f11= Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=init, name='conv8')(f10)

    x_class_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=init, name='conv9_class')(f11)

    # classifier

    x_class_2 = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid',  # + 1 for the background class
                       kernel_initializer=init_class, name="out_class")(x_class_1)

    #regressor
    # x_reg_1 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f9)
    # # x_reg_1_0 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='normal')(x_reg_1)
    #
    # x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear',
    #                  kernel_initializer='zeros', name="out_reg")(x_reg_1)
    #
    # model = Model(input_layer, [x_class_2, x_reg_2])
    model = Model(input_layer, [x_class_2])

    return model

from tensorflow.keras.layers import Conv2D, MaxPool2D, Input, concatenate,Cropping2D
from tensorflow.keras.models import Model, model_from_json

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


def construct_model_anchorless_detector_Pedestrians_anchorless_regressor(input_shape):
    """
    construct region proposal model architecture
    classifier architecture for binary classification
    :param input_shape: list of input dimensions (height, width, depth) [tuple]
    :param num_anchors: number of different anchors with the same center [int]
    :return:  model - Keras model object
    """
    #NOTE 56px receptive field so cekor 4(bez tretiot maxpooling

    #NOTE 84px receptive field so cekor 8

    input_layer = Input(shape=input_shape)


    # feature extractor
    f1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv1')(
        input_layer)
    f2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv2')(f1)
    f3 = MaxPool2D(pool_size=(2, 2))(f2)

    f4 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv4')(f3)
    f5 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv5')(f4)

    f6 = MaxPool2D(pool_size=(2, 2))(f5)

    f7 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv7')(f6)
    f8 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv8')(f7)
    f9 = MaxPool2D(pool_size=(2, 2))(f8)

    # f9=f8


    # f10 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f9)
    # f10_1 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f9)
    # f10_con=concatenate([f10_1, f10], axis=3)



    # f11 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f10)

    f10 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv10')(f9)
    f11= Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv11')(f10)

    x_class_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv12')(f11)

    # classifier

    x_class_2 = Conv2D(filters=2, kernel_size=(1, 1), padding='same', activation='softmax',  # + 1 for the background class
                       kernel_initializer='uniform', name="out_class")(x_class_1)

    #regressor
    x_reg_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv12_reg')(f11)
    # x_reg_1_0 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='normal')(x_reg_1)

    x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear',
                     kernel_initializer='zeros', name="out_reg")(x_reg_1)

    model = Model(input_layer, [x_class_2, x_reg_2])
    # model = Model(input_layer, [x_class_2])

    return model
def construct_model_anchorless_detector_Pedestrians_anchorless_padding_valid(input_shape):
    """
    construct region proposal model architecture
    classifier architecture for binary classification
    :param input_shape: list of input dimensions (height, width, depth) [tuple]
    :param num_anchors: number of different anchors with the same center [int]
    :return:  model - Keras model object
    """
    #NOTE 56px receptive field so cekor 4(bez tretiot maxpooling

    #NOTE 84px receptive field so cekor 8

    input_layer = Input(shape=input_shape)


    # feature extractor
    f1 = Conv2D(filters=32, kernel_size=(3, 3), padding='valid', activation='relu', kernel_initializer='normal')(
        input_layer)
    f2 = Conv2D(filters=32, kernel_size=(3, 3), padding='valid', activation='relu', kernel_initializer='normal')(f1)
    f3 = MaxPool2D(pool_size=(2, 2))(f2)

    f4 = Conv2D(filters=48, kernel_size=(3, 3), padding='valid', activation='relu', kernel_initializer='normal')(f3)
    f5 = Conv2D(filters=48, kernel_size=(3, 3), padding='valid', activation='relu', kernel_initializer='normal')(f4)

    f6 = MaxPool2D(pool_size=(2, 2))(f5)

    f7 = Conv2D(filters=48, kernel_size=(3, 3), padding='valid', activation='relu', kernel_initializer='normal')(f6)
    f8 = Conv2D(filters=48, kernel_size=(3, 3), padding='valid', activation='relu', kernel_initializer='normal')(f7)
    f9 = MaxPool2D(pool_size=(2, 2))(f8)

    # f9=f8


    # f10 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f9)
    # f10_1 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f9)
    # f10_con=concatenate([f10_1, f10], axis=3)



    # f11 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f10)

    f10 = Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', kernel_initializer='normal')(f9)
    f11= Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', kernel_initializer='normal')(f10)

    x_class_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', kernel_initializer='normal')(f11)

    # classifier

    x_class_2 = Conv2D(filters=2, kernel_size=(1, 1), padding='valid', activation='softmax',  # + 1 for the background class
                       kernel_initializer='uniform', name="out_class")(x_class_1)

    # regressor
    # x_reg_1 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f9)
    # # x_reg_1_0 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='normal')(x_reg_1)
    #
    # x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear',
    #                  kernel_initializer='zeros', name="out_reg")(x_reg_1)

    # model = Model(input_layer, [x_class_2, x_reg_2])
    model = Model(input_layer, [x_class_2])

    return model

def construct_model_ssd_pedestrians_ls_anchorfull(input_shape, num_anchors):
    """
    construct region proposal model architecture
    classifier architecture for binary classification
    :param input_shape: list of input dimensions (height, width, depth) [tuple]
    :param num_anchors: number of different anchors with the same center [int]
    :return:  model - Keras model object
    """

    input_layer = Input(shape=input_shape)

    # feature extractor
    f1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(
        input_layer)
    f2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f1)
    f3 = MaxPool2D(pool_size=(2, 2))(f2)

    f4 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f3)
    f5 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f4)
    f6 = MaxPool2D(pool_size=(2, 2))(f5)

    f7 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f6)
    f8 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f7)
    f9 = MaxPool2D(pool_size=(2, 2))(f8)

    f10 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f9)
    f11 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f10)
    # f12 = MaxPool2D(pool_size=(2, 2))(f11)

    # classifier
    x_class_1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu',
                       kernel_initializer='he_normal')(f11)
    x_class_2 = Conv2D(filters=num_anchors + 1, kernel_size=(1, 1), padding='same', activation='sigmoid',
                       kernel_initializer='glorot_uniform', name="rpn_out_class")(x_class_1)

    # regressor
    x_reg_1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f11)
    x_reg_2 = Conv2D(filters=num_anchors * 4, kernel_size=(1, 1), padding='same', activation='linear',
                     kernel_initializer='zeros', name="rpn_out_regress")(x_reg_1)

    model = Model(input_layer, [x_class_2, x_reg_2])

    return model
def construct_model_anchorless_detector_with_inceptions(input_shape):
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

    #otkomnentiraj psole

    f11_11 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)

    f_55=Conv2D(filters=48, kernel_size=(5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)

    f_con=concatenate([f11_11, f11, f_55], axis=3)

    #version2
    # f11_11= Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f10)
    # f_55=Conv2D(filters=48, kernel_size=(5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(f11_11)


    # classifier
    x_class_1 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f_con)
    x_class_2 = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='softmax',  # + 1 for the background class
                       kernel_initializer='uniform', name="out_class")(x_class_1)

    # regressor
    x_reg_1 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f_con)
    # x_reg_1_0 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='normal')(x_reg_1)

    x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear',
                     kernel_initializer='zeros', name="out_reg")(x_reg_1)

    model = Model(input_layer, [x_class_2, x_reg_2])

    return model

def construct_model_anchorless_detector_with_one_inception(input_shape):
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

    #otkomnentiraj psole

    # f11_11 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)
    #
    # f_55=Conv2D(filters=48, kernel_size=(5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)
    #
    # f_con=concatenate([f11_11, f11, f_55], axis=3)

    #version2
    # f11_11= Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f10)
    # f_55=Conv2D(filters=48, kernel_size=(5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(f11_11)


    # classifier
    x_class_1 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f11)
    x_class_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='softmax',  # + 1 for the background class
                       kernel_initializer='uniform', name="out_class")(x_class_1)

    # regressor
    x_reg_1 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f11)
    # x_reg_1_0 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='normal')(x_reg_1)

    x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear',
                     kernel_initializer='normal', name="out_reg")(x_reg_1)

    model = Model(input_layer, [x_class_2, x_reg_2])

    return model


def construct_model_anchorless_detector_skip_v1_half_filters_first_layer(input_shape):
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
    f1 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(input_layer)
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
    x_class_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f_con)
    x_class_2 = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='softmax',  # + 1 for the background class
                       kernel_initializer='uniform', name="out_class")(x_class_1)

    # regressor
    x_reg_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f_con)
    x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear', kernel_initializer='zeros', name="out_reg")(x_reg_1)

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
    f1 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(
        input_layer)
    f2 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f1)
    f3 = MaxPool2D(pool_size=(2, 2))(f2)

    f4 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f3)
    f5 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f4)
    f6 = MaxPool2D(pool_size=(2, 2))(f5)

    f7 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f6)
    f8 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f7)
    f9 = MaxPool2D(pool_size=(2, 2))(f8)

    concat_skip=f9

    # skip1 = Conv2D(filters=48, kernel_size=(5, 5), strides=(4, 4), padding='same', activation='relu', kernel_initializer='normal')(f3)
    # skip1= Cropping2D(cropping=((1, 0), (0, 0)))(skip1)
    # concat_skip = concatenate([skip1, f9])

    # inception 1:
    f10 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(concat_skip)
    # f10_1 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(concat_skip)
    # f10_con = concatenate([f10_1, f10], axis=3)
    f10_con=f10
    # inception 2:
    f11 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)
    f11_11 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)
    f_55 = Conv2D(filters=48, kernel_size=(5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)
    f_con = concatenate([f11_11, f11, f_55], axis=3)

    # classifier
    x_class_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f_con)
    x_class_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='softmax',  # + 1 for the background class
                       kernel_initializer='uniform', name="out_class")(x_class_1)

    # regressor
    x_reg_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f_con)
    x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear', kernel_initializer='zeros', name="out_reg")(x_reg_1)

    model = Model(input_layer, [x_class_2, x_reg_2])

    return model

def construct_model_anchorless_detector_skip_v1_custom_loss_base(input_shape):
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
    f1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(
        input_layer)
    f2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f1)
    f3 = MaxPool2D(pool_size=(2, 2))(f2)

    f4 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f3)
    f5 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f4)
    f6 = MaxPool2D(pool_size=(2, 2))(f5)

    f7 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f6)
    f8 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f7)
    f9 = MaxPool2D(pool_size=(2, 2))(f8)

    f10 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f9)
    f11 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f10)

    # skip1 = Conv2D(filters=48, kernel_size=(5, 5), strides=(4, 4), padding='same', activation='relu', kernel_initializer='normal')(f3)
    # skip1= Cropping2D(cropping=((0, 0), (0, 0)))(skip1)
    # concat_skip = concatenate([skip1, f9])

    # inception 1:
    # f10 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(concat_skip)
    # f10_1 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(concat_skip)
    # f10_con = concatenate([f10_1, f10], axis=3)

    # inception 2:
    # f11 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)
    # f11_11 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)
    # f_55 = Conv2D(filters=48, kernel_size=(5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)
    # f_con = concatenate([f11_11, f11, f_55], axis=3)

    # classifier
    x_class_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f11)
    x_class_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='softmax',  # + 1 for the background class
                       kernel_initializer='uniform', name="out_class")(x_class_1)


    # regressor
    x_reg_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f11)
    x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear', kernel_initializer='normal', name="out_reg")(x_reg_1)

    output_combined=concatenate([x_class_2,x_reg_2], axis=-1)
    model = Model(input_layer, output_combined)
    return model

import tensorflow
mirrored_strategy = tensorflow.distribute.MirroredStrategy(devices=["/job:localhost/replica:0/task:0/device:GPU:0", "/job:localhost/replica:0/task:0/device:GPU:1"],
                                                   cross_device_ops=tensorflow.distribute.HierarchicalCopyAllReduce())
# print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))


# with mirrored_strategy.scope():

def construct_model_anchorless_detector_skip_v1_custom_loss_skiponly(input_shape):
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

    skip1 = Conv2D(filters=48, kernel_size=(5, 5), strides=(4, 4), padding='same', activation='relu',
                   kernel_initializer='normal')(f3)
    skip1 = Cropping2D(cropping=((1, 0), (0, 0)))(skip1)
    concat_skip = concatenate([skip1, f9])

    # inception 1:
    f10 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(
        concat_skip)
    # f10_1 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(
    #     concat_skip)
    # f10_con = concatenate([f10_1, f10], axis=3)

    # inception 2:
    f11 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(
        f10)



    # classifier
    x_class_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(
        f11)
    x_class_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='softmax',
                       # + 1 for the background class
                       kernel_initializer='uniform', name="out_class")(x_class_1)

    # regressor
    x_reg_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(
        f11)
    x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear', kernel_initializer='normal',
                     name="out_reg")(x_reg_1)

    output_combined = concatenate([x_class_2, x_reg_2], axis=-1)
    model = Model(input_layer, output_combined)
    return model
with mirrored_strategy.scope():
    def construct_model_anchorless_detector_skip_v1_custom_loss(input_shape):
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

        # Compute dynamic crop in height
        # def crop_to_match(inputs):
        #     skip, f9 = inputs
        #     # Crop skip to match f9 height
        #     skip_height = tf.shape(skip)[1]
        #     f9_height = tf.shape(f9)[1]
        #     crop_top = skip_height - f9_height
        #     return skip[:, crop_top:, :, :]  # crop top rows
        #
        # # Apply Lambda
        # skip1_cropped = Lambda(crop_to_match)([skip1, f9])

        # concat_skip = concatenate([skip1, f9], axis=3)

        # skip1 = ZeroPadding2D(padding=((1, 0), (0, 0)))(skip1)
        #skip1 =Cropping2D(cropping=((1, 0), (0, 0)))(skip1)
        f9 = ZeroPadding2D(padding=((1, 0), (0, 0)))(f9)

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
        x_class_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f_con)
        x_class_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='softmax',  # + 1 for the background class
                           kernel_initializer='uniform', name="out_class")(x_class_1)


        # regressor
        x_reg_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f_con)
        x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear', kernel_initializer='normal', name="out_reg")(x_reg_1)

        output_combined=concatenate([x_class_2,x_reg_2], axis=-1)
        model = Model(input_layer, output_combined)

        return model

def cbam_block(feature_map, reduction_ratio=8, name="cbam"):
    """CBAM: Convolutional Block Attention Module"""
    channel = feature_map.shape[-1]

    # Channel Attention
    shared_layer_one = Dense(channel // reduction_ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(feature_map)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(feature_map)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    channel_refined = Multiply()([feature_map, cbam_feature])

    # Spatial Attention
    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True))(channel_refined)
    max_pool = Lambda(lambda x: tf.reduce_max(x, axis=3, keepdims=True))(channel_refined)
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    spatial = Conv2D(filters=1,
                     kernel_size=7,
                     strides=1,
                     padding='same',
                     activation='sigmoid',
                     kernel_initializer='he_normal',
                     use_bias=False)(concat)
    refined_feature = Multiply()([channel_refined, spatial])

    return refined_feature


with mirrored_strategy.scope():
    def construct_model_anchorless_detector_skip_v1_custom_loss_cbam(input_shape):
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
        skip1 =Cropping2D(cropping=((1, 0), (0, 0)))(skip1)
        concat_skip = concatenate([skip1, f9])
        concat_skip = cbam_block(concat_skip, name="cbam1")

        # inception 1:
        f10 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(concat_skip)
        f10_1 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(concat_skip)
        f10_con = concatenate([f10_1, f10], axis=3)

        # inception 2:
        f11 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)
        f11_11 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)
        f_55 = Conv2D(filters=48, kernel_size=(5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)
        f_con = concatenate([f11_11, f11, f_55], axis=3)

        f_con = cbam_block(f_con, name="cbam2")

        # classifier
        x_class_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f_con)
        x_class_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='softmax',  # + 1 for the background class
                           kernel_initializer='uniform', name="out_class")(x_class_1)


        # regressor
        x_reg_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f_con)
        x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear', kernel_initializer='normal', name="out_reg")(x_reg_1)

        output_combined=concatenate([x_class_2,x_reg_2], axis=-1)
        model = Model(input_layer, output_combined)

        return model
def construct_model_anchorless_detector_skip_v1_custom_loss_1inc(input_shape):
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

    # skip1 = Conv2D(filters=48, kernel_size=(5, 5), strides=(4, 4), padding='same', activation='relu', kernel_initializer='normal')(f3)
    # skip1= Cropping2D(cropping=((1, 0), (0, 0)))(skip1)
    # concat_skip = concatenate([skip1, f9])

    # inception 1:
    f10 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f9)
    f10_1 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f9)
    f10_con = concatenate([f10_1, f10], axis=3)

    # # inception 2:
    # f11 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)
    # f11_11 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)
    # f_55 = Conv2D(filters=48, kernel_size=(5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)
    # f_con = concatenate([f11_11, f11, f_55], axis=3)

    # classifier
    x_class_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f10_con)
    x_class_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='softmax',  # + 1 for the background class
                       kernel_initializer='uniform', name="out_class")(x_class_1)


    # regressor
    x_reg_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f10_con)
    x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear', kernel_initializer='normal', name="out_reg")(x_reg_1)

    output_combined=concatenate([x_class_2,x_reg_2], axis=-1)
    model = Model(input_layer, output_combined)

    return model
def construct_model_anchorless_detector_skip_v1_drug_loss(input_shape):
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
    x_class_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f_con)
    x_class_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='softmax',  # + 1 for the background class
                       kernel_initializer='uniform', name="out_class")(x_class_1)

    # regressor
    x_reg_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f_con)
    x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear', kernel_initializer='zeros', name="out_reg")(x_reg_1)

    Con=concatenate(x_class_2,x_reg_2)
    model = Model(input_layer, Con)

    return model

def construct_model_anchorless_detector_wavelet_simple(input_shape):
    """
    construct region proposal model architecture
    classifier architecture for binary classification
    :param input_shape: list of input dimensions (height, width, depth) [tuple]
    :param num_anchors: number of different anchors with the same center [int]
    :return:  model - Keras model object
    """
#&NOTE receptive field na vakva mreza: 46px
    input_layer = Input(shape=input_shape)

    #feature extractor
    f1 = Conv2D(filters=4, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(
        input_layer)
    f2 = Conv2D(filters=4, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f1)
    f3 = MaxPool2D(pool_size=(2, 2))(f2)
    # f3=input_layer
    f4 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f3)
    f5 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f4)
    f6 = MaxPool2D(pool_size=(2, 2))(f5)
        #86x128

    # f6= Cropping2D(cropping=((1, 0), (0, 0)))(f6)


    f7 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f6)
    f8 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f7)

    f9 = MaxPool2D(pool_size=(2, 2))(f8)
        #43x64
    # skip1 = Conv2D(filters=48, kernel_size=(5, 5), strides=(4, 4), padding='same', activation='relu', kernel_initializer='normal')(f3)
    # concat_skip = concatenate([skip1, f9])

    # inception 1:
    f10 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f9)
    # f10_1 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(concat_skip)
    # f10_con = concatenate([f10_1, f10], axis=3)

    # inception 2:
    f11 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f10)


    # classifier
    x_class_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f11)
    x_class_2 = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='softmax',  # + 1 for the background class
                       kernel_initializer='uniform', name="out_class")(x_class_1)

    # regressor
    x_reg_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f11)
    x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear', kernel_initializer='zeros', name="out_reg")(x_reg_1)

    model = Model(input_layer, [x_class_2, x_reg_2])

    return model

import numpy as np
import  tensorflow.keras.backend as K
#sobel kernel
def my_filter(shape, dtype=None):
        # filter=np.zeros((8,3,3))
    filter_sobel_vertical = np.array(
                    [[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])
    filter_sobel_vertical_flip=np.array(
                    [[1, 0, -1],
                     [2, 0, -2],
                     [1, 0, -1]])
    filter_sobel_horizontal=np.array(
                    [[1, 2, 1],
                     [0, 0, 0],
                     [-1, -2, -1]])

    filter_sobel_horizontal1 = np.array(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]])


    filter_sobel_diagonal1=np.array(
         [[0, 1, 2],
         [-1, 0, 1],
         [-2, -1, 0]]
        )

    filter_sobel_diagonal1_1 = np.array(
            [[0, -1, -2],
             [1, 0, -1],
             [2, 1, 0]]
        )


    filter_sobel_diagonal2=np.array(
        [[-2, -1, 0],
         [-1, 0, 1],
         [0, 1, 2]]
        )
    filter_sobel_diagonal2_2 = np.array(
            [[2, 1, 0],
             [1, 0, -1],
             [0, -1, -2]]
        )

    filter_sobel_vertical = filter_sobel_vertical.reshape((3, 3, 1))
    filter_sobel_vertical_flip = filter_sobel_vertical_flip.reshape((3, 3, 1))
    filter_sobel_horizontal = filter_sobel_horizontal.reshape((3, 3, 1))
    filter_sobel_horizontal1 = filter_sobel_horizontal1.reshape((3, 3, 1))
    filter_sobel_diagonal1 = filter_sobel_diagonal1.reshape((3, 3, 1))
    filter_sobel_diagonal1_1 = filter_sobel_diagonal1_1.reshape((3, 3, 1))
    filter_sobel_diagonal2 = filter_sobel_diagonal2.reshape((3, 3, 1))
    filter_sobel_diagonal2_2 = filter_sobel_diagonal2_2.reshape((3, 3, 1))
    import tensorflow as tf
    filter = np.concatenate([[filter_sobel_vertical,filter_sobel_vertical_flip,filter_sobel_horizontal,filter_sobel_horizontal1,filter_sobel_diagonal1,filter_sobel_diagonal1_1,filter_sobel_diagonal2,filter_sobel_diagonal2_2]], axis=0)
    filter = filter.transpose((1, 2, 3, 0))
    # filter=tf.convert_to_tensor(filter, dtype='float32')

    # assert filter.shape==shape
    # return K.variable(filter, dtype='float32')
    return filter


def my_filter_2(shape, dtype=None):
        # filter=np.zeros((8,3,3))
    filter_sobel_vertical = np.array(
                    [[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])
    filter_sobel_vertical_flip=np.array(
                    [[1, 0, -1],
                     [2, 0, -2],
                     [1, 0, -1]])
    filter_sobel_horizontal=np.array(
                    [[1, 2, 1],
                     [0, 0, 0],
                     [-1, -2, -1]])

    filter_sobel_horizontal1 = np.array(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]])


    filter_sobel_diagonal1=np.array(
         [[0, 1, 2],
         [-1, 0, 1],
         [-2, -1, 0]]
        )

    filter_sobel_diagonal1_1 = np.array(
            [[0, -1, -2],
             [1, 0, -1],
             [2, 1, 0]]
        )


    filter_sobel_diagonal2=np.array(
        [[-2, -1, 0],
         [-1, 0, 1],
         [0, 1, 2]]
        )
    filter_sobel_diagonal2_2 = np.array(
            [[2, 1, 0],
             [1, 0, -1],
             [0, -1, -2]]
        )
    filter_sobel_vertical= np.repeat(filter_sobel_vertical[:, :, np.newaxis], 32, axis=2)
    filter_sobel_vertical_flip= np.repeat(filter_sobel_vertical_flip[:, :, np.newaxis], 32, axis=2)
    filter_sobel_horizontal= np.repeat(filter_sobel_horizontal[:, :, np.newaxis], 32, axis=2)
    filter_sobel_horizontal1= np.repeat(filter_sobel_horizontal1[:, :, np.newaxis], 32, axis=2)
    filter_sobel_diagonal1= np.repeat(filter_sobel_diagonal1[:, :, np.newaxis], 32, axis=2)
    filter_sobel_diagonal1_1= np.repeat(filter_sobel_diagonal1_1[:, :, np.newaxis], 32, axis=2)
    filter_sobel_diagonal2= np.repeat(filter_sobel_diagonal2[:, :, np.newaxis], 32, axis=2)
    filter_sobel_diagonal2_2= np.repeat(filter_sobel_diagonal2_2[:, :, np.newaxis], 32, axis=2)
    #
    # filter_sobel_vertical = filter_sobel_vertical.reshape((3, 3, 1))
    # filter_sobel_vertical_flip = filter_sobel_vertical_flip.reshape((3, 3, 1))
    # filter_sobel_horizontal = filter_sobel_horizontal.reshape((3, 3, 1))
    # filter_sobel_horizontal1 = filter_sobel_horizontal1.reshape((3, 3, 1))
    # filter_sobel_diagonal1 = filter_sobel_diagonal1.reshape((3, 3, 1))
    # filter_sobel_diagonal1_1 = filter_sobel_diagonal1_1.reshape((3, 3, 1))
    # filter_sobel_diagonal2 = filter_sobel_diagonal2.reshape((3, 3, 1))
    # filter_sobel_diagonal2_2 = filter_sobel_diagonal2_2.reshape((3, 3, 1))
    import tensorflow as tf
    filter = np.concatenate([[filter_sobel_vertical,filter_sobel_vertical_flip,filter_sobel_horizontal,filter_sobel_horizontal1,filter_sobel_diagonal1,filter_sobel_diagonal1_1,filter_sobel_diagonal2,filter_sobel_diagonal2_2]], axis=0)
    filter = filter.transpose((1, 2, 3, 0))
    # filter=tf.convert_to_tensor(filter, dtype='float32')
    # print(1)
    # assert filter.shape==shape
    # return K.variable(filter, dtype='float32')
    return filter

# my_filter_2((3,3,32,8), dtype=None)
# print(10)




def construct_model_anchorless_detector_skip_v1_fixed_filters(input_shape):
    """
    construct region proposal model architecture
    classifier architecture for binary classification
    :param input_shape: list of input dimensions (height, width, depth) [tuple]
    :param num_anchors: number of different anchors with the same center [int]
    :return:  model - Keras model object
    """

#&NOTE receptive field na vakva mreza: 46px
    input_layer = Input(shape=input_shape)

    # in = Input(....)

    filters1=8
    filters2=24
    f1_p1 = Conv2D(filters=filters1, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(input_layer)
    f1_p2 = Conv2D(filters=filters2, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(input_layer)

    # f1_p11=f1_p1(input_layer)
    # f1_p22=f1_p2(input_layer)

    # convB.set_weights(some_weight_array)

    # conv1 = convA( in)
    # conv2 = convB( in)

    f1 = concatenate([f1_p1, f1_p2], axis=-1)

    # feature extractor
    # f1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(
    #     input_layer)
    f2_p1 = Conv2D(filters=filters1, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f1)
    f2_p2 = Conv2D(filters=filters2, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f1)
    f2 = concatenate([f2_p1, f2_p2], axis=-1)

    #f2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f1)
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
    x_class_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f_con)
    x_class_2 = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='softmax',  # + 1 for the background class
                       kernel_initializer='uniform', name="out_class")(x_class_1)

    # regressor
    x_reg_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f_con)
    x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear', kernel_initializer='zeros', name="out_reg")(x_reg_1)

    model = Model(input_layer, [x_class_2, x_reg_2])

    return model


# import pywt
import tensorflow as tf
import tensorflow.keras as keras


def wavelet(data):
    # f_maps=[]
    # res=tf.map_fn(lambda t: t, data)
    # print(tf.shape(res))
    # print(1)
    # for x in data:
    # coeffs2=pywt.dwt2(x[:,:,0], 'db1')
    # LL, (LH, HL, HH) = coeffs2
    # LL = LL.reshape(LL.shape + (1,))
    # LH = LH.reshape(LH.shape + (1,))
    # HL = HL.reshape(HL.shape + (1,))
    # HH = HH.reshape(HH.shape + (1,))
    # sliki = np.concatenate([LL, LH, HL, HH], axis=-1)
    # f_maps.append(sliki)
    return None
#
# def wavelet_transform(inputs):
#         #a = inputs.eval(session=tf.compat.v1.Session())
#
#         #a = inputs.eval(session=tf.Session())
#         a = tf.Session().run(inputs)
#
#         cA = np.zeros((inputs.shape[0], inputs.shape[1], inputs.shape[2]))
#         cH = np.zeros((inputs.shape[0], inputs.shape[1], inputs.shape[2]))
#         cV = np.zeros((inputs.shape[0], inputs.shape[1], inputs.shape[2]))
#         cD = np.zeros((inputs.shape[0], inputs.shape[1], inputs.shape[2]))
#         for i in range(a.shape[0]):
#             image = inputs[i, :, :, :]
#             coeffs = pywt.dwt2(image, 'db1')
#             cA[i, :, :], (cH[i, :, :], cV[i, :, :], cD[i, :, :]) = coeffs
#
#
#         return np.concatenate([cA[:, :, :, np.newaxis], cH[:, :, :, np.newaxis], cV[:, :, :, np.newaxis], cD[:, :, :, np.newaxis]], axis=-1)

class WaveletTransformLayer(keras.layers.Layer):
    def __init__(self, wavelet_name, level=1, **kwargs):
        super(WaveletTransformLayer, self).__init__(**kwargs)
        self.wavelet_name = wavelet_name
        self.level = level

    def build(self, input_shape):
        self.coeffs = None
        super(WaveletTransformLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        # inputs = tf.reshape(inputs, [-1, 341 * 512 * 1])
        coeffs = pywt.wavedecn(inputs, self.wavelet_name, level=self.level)
        self.coeffs = coeffs
        return coeffs[0]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 341 * 512 * 1)

def construct_model_anchorless_detector_skip_v1_wavelets_new1(input_shape):
    """
    construct region proposal model architecture
    classifier architecture for binary classification
    :param input_shape: list of input dimensions (height, width, depth) [tuple]
    :param num_anchors: number of different anchors with the same center [int]
    :return:  model - Keras model object
    """
    import keras.backend as K
    K.clear_session()
    import pywt
    import pywt.data
    from matplotlib import pyplot as plt
#&NOTE receptive field na vakva mreza: 46px



    input_layer = Input(shape=input_shape)

    # in = Input(....)

    # filters1=8
    filters2=24


    #f1_p1 = Conv2D(filters=filters1, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(input_layer)
    print(input_layer.shape)

    # for i in range(input_layer.shape[0]):
    #     f1_p1=coeffs2 = pywt.dwt2(input_layer[i], 'bior1.3')
    #     LL, (LH, HL, HH) = coeffs2
    #     fig = plt.figure(figsize=(12, 3))
    #     for i, a in enumerate([LL, LH, HL, HH]):
    #         ax = fig.add_subplot(1, 4, i + 1)
    #         ax.imshow(a, interpolation="nearest", cmap='gray')
    #         # ax.set_title(titles[i], fontsize=10)
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #
    #     fig.tight_layout()
    #     plt.show()
    #
    # f1_pom=concatenate([LL,LH,HL,HH], axis=-1)

    # f1_p2 = Conv2D(filters=filters2, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(input_layer)

    # f1_p11=f1_p1(input_layer)
    # f1_p22=f1_p2(input_layer)

    # convB.set_weights(some_weight_array)

    # conv1 = convA( in)
    # conv2 = convB( in)
    # f1=f1_p2
    # f1 = concatenate([f1_p1, f1_p2], axis=-1)

    # feature extractor
    # f1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(
    #     input_layer)
    # f2_p1 = Conv2D(filters=filters1, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f1)
    # f2_p2 = Conv2D(filters=filters2, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f1)
    # f2 = concatenate([f2_p1, f2_p2], axis=-1)
    # def wavelet_transform(inputs):
    #     inputs = inputs.numpy()
    #     output = np.zeros((inputs.shape[0], inputs.shape[1], inputs.shape[2], 4))
    #     for i in range(inputs.shape[0]):
    #         image = inputs[i, :, :, :]
    #         coeffs = pywt.dwt2(image, 'db1')
    #         cA, (cH, cV, cD) = coeffs
    #         output[i, :, :, 0] = cA
    #         output[i, :, :, 1] = cH
    #         output[i, :, :, 2] = cV
    #         output[i, :, :, 3] = cD
    #     return output



    # f1 = tf.keras.layers.Lambda(wavelet_transform)(input_layer)

    f1 = tf.keras.layers.Lambda(wavelet_transform)(input_layer)
    f1 = tf.keras.layers.Reshape((input_shape[0], input_shape[1], 4))(f1)
    #f1 = tf.keras.layers.Lambda(wavelet)(input_layer)


    f2_1=Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f1)

    f2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f2_1)
    # f3 = MaxPool2D(pool_size=(2, 2))(f1_p2)
    f3=f2
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
    x_class_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f_con)
    x_class_2 = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='softmax',  # + 1 for the background class
                       kernel_initializer='uniform', name="out_class")(x_class_1)

    # regressor
    x_reg_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f_con)
    x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear', kernel_initializer='zeros', name="out_reg")(x_reg_1)

    model = Model(input_layer, [x_class_2, x_reg_2])

    return model


def construct_model_anchorless_detector_skip_v1_wavelets_new(input_shape):
    """
    construct region proposal model architecture
    classifier architecture for binary classification
    :param input_shape: list of input dimensions (height, width, depth) [tuple]
    :param num_anchors: number of different anchors with the same center [int]
    :return:  model - Keras model object
    """
    import keras.backend as K
    # K.clear_session()
    import pywt
    import pywt.data
    from matplotlib import pyplot as plt
#&NOTE receptive field na vakva mreza: 46px



    # input_layer = Input(shape=input_shape)



    # Define the input layer
    inputs = keras.Input(shape=input_shape)

    # Perform some operations on the input tensors
    f1 = WaveletTransformLayer("db4", level=1)(inputs)
    # in = Input(....)

    # filters1=8
    filters2=24


    #f1_p1 = Conv2D(filters=filters1, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(input_layer)
    print(inputs.shape)

    # for i in range(input_layer.shape[0]):
    #     f1_p1=coeffs2 = pywt.dwt2(input_layer[i], 'bior1.3')
    #     LL, (LH, HL, HH) = coeffs2
    #     fig = plt.figure(figsize=(12, 3))
    #     for i, a in enumerate([LL, LH, HL, HH]):
    #         ax = fig.add_subplot(1, 4, i + 1)
    #         ax.imshow(a, interpolation="nearest", cmap='gray')
    #         # ax.set_title(titles[i], fontsize=10)
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #
    #     fig.tight_layout()
    #     plt.show()
    #
    # f1_pom=concatenate([LL,LH,HL,HH], axis=-1)

    # f1_p2 = Conv2D(filters=filters2, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(input_layer)

    # f1_p11=f1_p1(input_layer)
    # f1_p22=f1_p2(input_layer)

    # convB.set_weights(some_weight_array)

    # conv1 = convA( in)
    # conv2 = convB( in)
    # f1=f1_p2
    # f1 = concatenate([f1_p1, f1_p2], axis=-1)

    # feature extractor
    # f1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(
    #     input_layer)
    # f2_p1 = Conv2D(filters=filters1, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f1)
    # f2_p2 = Conv2D(filters=filters2, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f1)
    # f2 = concatenate([f2_p1, f2_p2], axis=-1)
    # def wavelet_transform(inputs):
    #     inputs = inputs.numpy()
    #     output = np.zeros((inputs.shape[0], inputs.shape[1], inputs.shape[2], 4))
    #     for i in range(inputs.shape[0]):
    #         image = inputs[i, :, :, :]
    #         coeffs = pywt.dwt2(image, 'db1')
    #         cA, (cH, cV, cD) = coeffs
    #         output[i, :, :, 0] = cA
    #         output[i, :, :, 1] = cH
    #         output[i, :, :, 2] = cV
    #         output[i, :, :, 3] = cD
    #     return output



    # f1 = tf.keras.layers.Lambda(wavelet_transform)(input_layer)

    # f1 = tf.keras.layers.Lambda(wavelet_transform)(input_layer)
    #f1 = tf.keras.layers.Reshape((input_shape[0], input_shape[1], 4))(f1)
    #f1 = tf.keras.layers.Lambda(wavelet)(input_layer)


    f2_1=Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f1)

    f2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f2_1)
    # f3 = MaxPool2D(pool_size=(2, 2))(f1_p2)
    f3=f2
    f4 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f3)
    f5 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f4)
    f6 = MaxPool2D(pool_size=(2, 2))(f5)

    f7 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f6)
    f8 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f7)
    f9 = MaxPool2D(pool_size=(2, 2))(f8)

    #skip1 = Conv2D(filters=48, kernel_size=(5, 5), strides=(4, 4), padding='same', activation='relu', kernel_initializer='normal')(f3)
    #skip1= Cropping2D(cropping=((1, 0), (0, 0)))(skip1)
    #concat_skip = concatenate([skip1, f9])

    # inception 1:
    # f10 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(concat_skip)
    # f10_1 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(concat_skip)
    # f10_con = concatenate([f10_1, f10], axis=3)

    # inception 2:
    # f11 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)
    # f11_11 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)
    # f_55 = Conv2D(filters=48, kernel_size=(5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(f10_con)
    # f_con = concatenate([f11_11, f11, f_55], axis=3)

    # classifier
    x_class_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f9)
    x_class_2 = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='softmax',  # + 1 for the background class
                       kernel_initializer='uniform', name="out_class")(x_class_1)

    # regressor
    x_reg_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f9)
    x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear', kernel_initializer='zeros', name="out_reg")(x_reg_1)

    model = Model(input_layer, [x_class_2, x_reg_2])

    return model



def construct_model_anchorless_detector_skip_v1_wavelets(input_shape):
    """
    construct region proposal model architecture
    classifier architecture for binary classification
    :param input_shape: list of input dimensions (height, width, depth) [tuple]
    :param num_anchors: number of different anchors with the same center [int]
    :return:  model - Keras model object
    """
    import pywt
    import pywt.data
    from matplotlib import pyplot as plt
#&NOTE receptive field na vakva mreza: 46px
    input_layer = Input(shape=input_shape)

    # in = Input(....)

    # filters1=8
    filters2=24


    #f1_p1 = Conv2D(filters=filters1, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(input_layer)
    print(input_layer.shape)

    # for i in range(input_layer.shape[0]):
    #     f1_p1=coeffs2 = pywt.dwt2(input_layer[i], 'bior1.3')
    #     LL, (LH, HL, HH) = coeffs2
    #     fig = plt.figure(figsize=(12, 3))
    #     for i, a in enumerate([LL, LH, HL, HH]):
    #         ax = fig.add_subplot(1, 4, i + 1)
    #         ax.imshow(a, interpolation="nearest", cmap='gray')
    #         # ax.set_title(titles[i], fontsize=10)
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #
    #     fig.tight_layout()
    #     plt.show()
    #
    # f1_pom=concatenate([LL,LH,HL,HH], axis=-1)

    # f1_p2 = Conv2D(filters=filters2, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(input_layer)

    # f1_p11=f1_p1(input_layer)
    # f1_p22=f1_p2(input_layer)

    # convB.set_weights(some_weight_array)

    # conv1 = convA( in)
    # conv2 = convB( in)
    # f1=f1_p2
    # f1 = concatenate([f1_p1, f1_p2], axis=-1)

    # feature extractor
    # f1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(
    #     input_layer)
    # f2_p1 = Conv2D(filters=filters1, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f1)
    # f2_p2 = Conv2D(filters=filters2, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f1)
    # f2 = concatenate([f2_p1, f2_p2], axis=-1)

    f2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(input_layer)
    f3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f2)

    # f3 = MaxPool2D(pool_size=(2, 2))(f1_p2)
    # f3=f2
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
    x_class_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f_con)
    x_class_2 = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='softmax',  # + 1 for the background class
                       kernel_initializer='uniform', name="out_class")(x_class_1)

    # regressor
    x_reg_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f_con)
    x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear', kernel_initializer='zeros', name="out_reg")(x_reg_1)

    model = Model(input_layer, [x_class_2, x_reg_2])

    return model

def construct_model_anchorless_detector_skip_v1_centerness(input_shape):
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
    x_class_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f_con)

    x_centerness= Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid',
                       kernel_initializer='uniform', name="out_centerness")(x_class_1)

    x_class_2 = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='softmax',  # + 1 for the background class
                       kernel_initializer='uniform', name="out_class")(x_class_1)

    # regressor
    x_reg_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal')(f_con)
    x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear', kernel_initializer='zeros', name="out_reg")(x_reg_1)

    model = Model(input_layer, [x_class_2, x_centerness, x_reg_2])

    return model



def construct_model_anchorless_detector_cls(input_shape):
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

    f10 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f9)
    f11 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f10)

    # classifier
    x_class_1 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu',
                       kernel_initializer='normal')(f11)
    x_class_2 = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='sigmoid',  # + 1 for the background class
                       kernel_initializer='uniform', name="out_class")(x_class_1)

    model = Model(input_layer, x_class_2)

    return model


def repack_as_single_gpu(model_prev, func_construct_model, input_dims,i,finetune):
    """

    :param model_prev: multi_gpu model [Keras Model object]
    :param func_construct_model: function to construct model architecture
    :param input_dims: input dimensions [tuple]
    :return:
    """

    model_single = func_construct_model(input_shape=input_dims)
    if finetune:
        model_layer = model_prev.get_layer(name='model_3')
    else:
        model_layer = model_prev.get_layer(name='model_'+str(i))
    weights_biases = model_layer.get_weights()

    cnt = 0
    for i in range(len(model_single.layers)):
        if len(model_single.layers[i].trainable_weights) > 0:
            model_single.layers[i].set_weights([weights_biases[cnt], weights_biases[cnt + 1]])
            cnt += 2

    # print(model_single.summary)

    return model_single
'''
   inputs = Input(input_shape)
    
    # Wavelet Transformation Layer
    x = Lambda(lambda x: dwt2(x, 'db1'))(inputs)
    
    # Split the 4 feature maps from the wavelet transformation
    cA, (cH, cV, cD) = x[0], x[1:]

'''