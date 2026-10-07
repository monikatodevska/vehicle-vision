"""
Course: Mashinski vid, FEEIT, Spring 2021
Date: 09.03.2021

Description: function library
             loss functions: SSD classifier (one-hot encoded data), regressor
Python version: 3.6
"""

# python imports
from tensorflow.keras import backend as K

from tensorflow.keras.losses import categorical_crossentropy as categorical_crossentropy, binary_crossentropy as binary_crossentropy
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
# loss function hyperparameters - weight of each loss function
lambda_rpn_reg = 0.01
lambda_rpn_class = 1.0
mse = tf.keras.losses.MeanSquaredError()
anchor_stride=8
import numpy as np
import helper_losses


def iou_loss(y_true, y_pred):
    # iou loss for bounding box prediction
    # input must be as [x1, y1, x2, y2]

    # AOG = Area of Groundtruth box
    AoG = K.abs(K.transpose(y_true)[2] - K.transpose(y_true)[0] + 1) * K.abs(K.transpose(y_true)[3] - K.transpose(y_true)[1] + 1)

    # AOP = Area of Predicted box
    AoP = K.abs(K.transpose(y_pred)[2] - K.transpose(y_pred)[0] + 1) * K.abs(K.transpose(y_pred)[3] - K.transpose(y_pred)[1] + 1)

    # overlaps are the co-ordinates of intersection box
    overlap_0 = K.maximum(K.transpose(y_true)[0], K.transpose(y_pred)[0])
    overlap_1 = K.maximum(K.transpose(y_true)[1], K.transpose(y_pred)[1])
    overlap_2 = K.minimum(K.transpose(y_true)[2], K.transpose(y_pred)[2])
    overlap_3 = K.minimum(K.transpose(y_true)[3], K.transpose(y_pred)[3])

    # intersection area
    intersection = (overlap_2 - overlap_0 + 1) * (overlap_3 - overlap_1 + 1)

    # area of union of both boxes
    union = AoG + AoP - intersection

    # iou calculation
    iou = intersection / union

    # bounding values of iou to (0,1)
    iou = K.clip(iou, 0.0 + K.epsilon(), 1.0 - K.epsilon())

    # loss for the iou value
    iou_loss = -K.log(iou)

    return iou_loss
def custom_loss_class_and_reg(y_true, y_pred):

    y_true_reg=[]
    y_pred_reg=[]
    start_r = np.int(np.round(anchor_stride / 2))
    start_c = np.int(np.round(anchor_stride / 2))

    (out_class_true,out_reg_true)=tf.split(y_true, num_or_size_splits=2, axis=1)
    (out_class_pred, out_reg_pred)=tf.split(y_pred, num_or_size_splits=2, axis=1)

    # mask = K.cast(K.not_equal(out_class_true, 0), 'float32')
    r, c, d = np.nonzero(out_class_true == 1) #Note dali ovde treba od out class predicted?


    for im_ind, im in enumerate(out_class_true):
        y_true_reg_tmp=[]
        for pred_ind in range(len(r)):  # iterate over positive predictions

            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # pomestuvanja na granicite na bboxot
            min_r = max(0, int(center_row - out_reg_true[im_ind,r[pred_ind], c[pred_ind], 0]))
            min_c = max(0, int(center_col - out_reg_true[im_ind, r[pred_ind], c[pred_ind], 1]))
            max_r = max(0, int(center_row + out_reg_true[im_ind, r[pred_ind], c[pred_ind], 2]))
            max_c = max(0, int(center_col + out_reg_true[im_ind, r[pred_ind], c[pred_ind], 3]))

            y_true_reg_tmp.append([min_c,min_r,max_c,max_r])
        y_true_reg.append(y_true_reg_tmp)

    for im_ind, im in enumerate(out_class_pred):
        y_pred_reg_tmp = []
        for pred_ind in range(len(r)):  # iterate over positive predictions

            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # pomestuvanja na granicite na bboxot
            min_r = max(0, int(center_row - out_reg_true[im_ind, r[pred_ind], c[pred_ind], 0]))
            min_c = max(0, int(center_col - out_reg_true[im_ind, r[pred_ind], c[pred_ind], 1]))
            max_r = max(0, int(center_row + out_reg_true[im_ind, r[pred_ind], c[pred_ind], 2]))
            max_c = max(0, int(center_col + out_reg_true[im_ind, r[pred_ind], c[pred_ind], 3]))

            y_pred_reg_tmp.append([min_c, min_r, max_c, max_r])
        y_pred_reg.append(y_pred_reg_tmp)

    iou_loss(y_true_reg, y_pred_reg)




    return custom_loss_class_and_reg


def custom_loss_class_and_reg1(y_true, y_pred):
    y_true_reg=[]
    y_pred_reg=[]
    start_r = np.int(np.round(anchor_stride / 2))
    start_c = np.int(np.round(anchor_stride / 2))

    (out_class_true,out_reg_true)=tf.split(y_true, num_or_size_splits=2, axis=2)
    (out_class_pred, out_reg_pred)=tf.split(y_pred, num_or_size_splits=2, axis=2)

    for batch_idx in range(tf.shape(out_class_true)[0]):
        r, c, d = tf.unravel_index(tf.where(tf.equal(out_class_true[batch_idx], 1)), tf.shape(out_class_true)[1:3])

        y_true_reg_tmp=[]
        for pred_ind in range(tf.shape(r)[0]):  # iterate over positive predictions

            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # pomestuvanja na granicite na bboxot
            min_r = tf.maximum(0, tf.cast(center_row - out_reg_true[batch_idx, r[pred_ind], c[pred_ind], 0], tf.int32))
            min_c = tf.maximum(0, tf.cast(center_col - out_reg_true[batch_idx, r[pred_ind], c[pred_ind], 1], tf.int32))
            max_r = tf.maximum(0, tf.cast(center_row + out_reg_true[batch_idx, r[pred_ind], c[pred_ind], 2], tf.int32))
            max_c = tf.maximum(0, tf.cast(center_col + out_reg_true[batch_idx, r[pred_ind], c[pred_ind], 3], tf.int32))

            y_true_reg_tmp.append([min_c, min_r, max_c, max_r])
        y_true_reg.append(y_true_reg_tmp)

        y_pred_reg_tmp = []
        for pred_ind in range(tf.shape(r)[0]):  # iterate over positive predictions

            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # pomestuvanja na granicite na bboxot
            min_r = tf.maximum(0, tf.cast(center_row - out_reg_pred[batch_idx, r[pred_ind], c[pred_ind], 0], tf.int32))
            min_c = tf.maximum(0, tf.cast(center_col - out_reg_pred[batch_idx, r[pred_ind], c[pred_ind], 1], tf.int32))
            max_r = tf.maximum(0, tf.cast(center_row + out_reg_pred[batch_idx, r[pred_ind], c[pred_ind], 2], tf.int32))
            max_c = tf.maximum(0, tf.cast(center_col + out_reg_pred[batch_idx, r[pred_ind], c[pred_ind], 3], tf.int32))

            y_pred_reg_tmp.append([min_c, min_r, max_c, max_r])
        y_pred_reg.append(y_pred_reg_tmp)

    return iou_loss1(y_true_reg, y_pred_reg)

import tensorflow as tf
def custom_loss_class_and_reg2(y_true, y_pred):
    y_true_reg=[]
    y_pred_reg=[]
    start_r = np.int(np.round(anchor_stride / 2))
    start_c = np.int(np.round(anchor_stride / 2))

    (out_class_true,out_reg_true)=(tf.split(y_true, num_or_size_splits=2, axis=2))
    (out_class_pred, out_reg_pred)=(tf.split(y_pred, num_or_size_splits=2, axis=2))

    # b=out_class_true.eval(session=tf.compat.v1.Session())
    # print(b)
    print(out_class_true)
    print(out_class_true.shape)

    for batch_idx in range(tf.shape(out_class_true)[0]):
        r, c, d = tf.unravel_index(tf.where(tf.equal(out_class_true[batch_idx], 1)), tf.shape(out_class_true)[1:3])

        y_true_reg_tmp=[]
        for pred_ind in range(tf.shape(r)[0]):  # iterate over positive predictions

            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # pomestuvanja na granicite na bboxot
            min_r = tf.maximum(0, tf.cast(center_row - out_reg_true[batch_idx, r[pred_ind], c[pred_ind], 0], tf.int32))
            min_c = tf.maximum(0, tf.cast(center_col - out_reg_true[batch_idx, r[pred_ind], c[pred_ind], 1], tf.int32))
            max_r = tf.maximum(0, tf.cast(center_row + out_reg_true[batch_idx, r[pred_ind], c[pred_ind], 2], tf.int32))
            max_c = tf.maximum(0, tf.cast(center_col + out_reg_true[batch_idx, r[pred_ind], c[pred_ind], 3], tf.int32))

            y_true_reg_tmp.append([min_c, min_r, max_c, max_r])
        y_true_reg.append(y_true_reg_tmp)

        y_pred_reg_tmp = []
        for pred_ind in range(tf.shape(r)[0]):  # iterate over positive predictions

            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # pomestuvanja na granicite na bboxot
            min_r = tf.maximum(0, tf.cast(center_row - out_reg_pred[batch_idx, r[pred_ind], c[pred_ind], 0], tf.int32))
            min_c = tf.maximum(0, tf.cast(center_col - out_reg_pred[batch_idx, r[pred_ind], c[pred_ind], 1], tf.int32))
            max_r = tf.maximum(0, tf.cast(center_row + out_reg_pred[batch_idx, r[pred_ind], c[pred_ind], 2], tf.int32))
            max_c = tf.maximum(0, tf.cast(center_col + out_reg_pred[batch_idx, r[pred_ind], c[pred_ind], 3], tf.int32))

            y_pred_reg_tmp.append([min_c, min_r, max_c, max_r])
        y_pred_reg.append(y_pred_reg_tmp)

    return iou_loss2(y_true_reg, y_pred_reg) + rpn_loss_cls_new(out_class_true,out_class_pred)

def iou_loss2(y_true, y_pred):
    # iou loss for bounding box prediction
    # input must be as [batch_size, [x1, y1, x2, y2]]

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # AOG = Area of Groundtruth box
    AoG = tf.abs(y_true[:, :, 2] - y_true[:, :, 0] + 1) * tf.abs(y_true[:, :, 3] - y_true[:, :, 1] + 1)

    # AOP = Area of Predicted box
    AoP = tf.abs(y_pred[:, :, 2] - y_pred[:, :, 0] + 1) * tf.abs(y_pred[:, :, 3] - y_pred[:, :, 1] + 1)

    # overlaps are the co-ordinates of intersection box
    overlap_0 = tf.maximum(y_true[:, :, 0], y_pred[:, :, 0])
    overlap_1 = tf.maximum(y_true[:, :, 1], y_pred[:, :, 1])
    overlap_2 = tf.minimum(y_true[:, :, 2], y_pred[:, :, 2])
    overlap_3 = tf.minimum(y_true[:, :, 3], y_pred[:, :, 3])

    # intersection area
    intersection = tf.maximum(overlap_2 - overlap_0 + 1, 0) * tf.maximum(overlap_3 - overlap_1 + 1, 0)

    # area of union of both boxes
    union = AoG + AoP - intersection

    # iou calculation
    iou = intersection / union

    # bounding values of iou to (0,1)
    iou = tf.clip_by_value(iou, 0.0 + K.epsilon(), 1.0 - K.epsilon())

    # loss for the iou value
    iou_loss = -tf.math.log(iou)

    # mean iou loss over batch
    mean_iou_loss = tf.reduce_mean(iou_loss)

    return mean_iou_loss
def iou_loss1(y_true, y_pred):
    # iou loss for bounding box prediction
    # input must be as [batch_size, M, [x1, y1, x2, y2]]

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # get number of bounding boxes for each sample in batch
    n_boxes = tf.shape(y_true)[1]

    # AOG = Area of Groundtruth boxes
    AoG = tf.reduce_prod(tf.maximum(y_true[:, :, 2:] - y_true[:, :, :2] + 1, 0), axis=-1)

    # AOP = Area of Predicted boxes
    AoP = tf.reduce_prod(tf.maximum(y_pred[:, :, 2:] - y_pred[:, :, :2] + 1, 0), axis=-1)

    # overlaps are the co-ordinates of intersection boxes
    overlap_0 = tf.maximum(tf.tile(tf.expand_dims(y_true[:, :, 0], axis=-1), [1, 1, n_boxes]),
                            tf.tile(tf.expand_dims(y_pred[:, :, 0], axis=-2), [1, n_boxes, 1]))
    overlap_1 = tf.maximum(tf.tile(tf.expand_dims(y_true[:, :, 1], axis=-1), [1, 1, n_boxes]),
                            tf.tile(tf.expand_dims(y_pred[:, :, 1], axis=-2), [1, n_boxes, 1]))
    overlap_2 = tf.minimum(tf.tile(tf.expand_dims(y_true[:, :, 2], axis=-1), [1, 1, n_boxes]),
                            tf.tile(tf.expand_dims(y_pred[:, :, 2], axis=-2), [1, n_boxes, 1]))
    overlap_3 = tf.minimum(tf.tile(tf.expand_dims(y_true[:, :, 3], axis=-1), [1, 1, n_boxes]),
                            tf.tile(tf.expand_dims(y_pred[:, :, 3], axis=-2), [1, n_boxes, 1]))

    # intersection area
    intersection = tf.reduce_prod(tf.maximum(overlap_2 - overlap_0 + 1, 0) * tf.maximum(overlap_3 - overlap_1 + 1, 0), axis=-1)

    # area of union of both boxes
    union = AoG + AoP - intersection

    # iou calculation
    iou = intersection / union

    # bounding values of iou to (0,1)
    iou = tf.clip_by_value(iou, 0.0 + K.epsilon(), 1.0 - K.epsilon())

    # loss for the iou value
    iou_loss = -tf.math.log(iou)

    # mean iou loss over batch
    mean_iou_loss = tf.reduce_mean(tf.reduce_sum(iou_loss, axis=-1) / tf.cast(n_boxes, tf.float32))

    return mean_iou_loss

def rpn_loss_reg(y_true, y_pred):
    """
    Leaky L1 norm loss, adaptated to exclude ignored pixels
    :param y_true: ground truth output [ndarray]
    :param y_pred: predicted output [ndarray]
    :return: loss function value []
    """

    mask = K.cast(K.not_equal(y_true, 0), 'float32')

    aaa = y_true - y_pred
    aaa = aaa*mask

    x_abs = K.abs(aaa)
    x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')

    # return lambda_rpn_reg * K.sum(x_bool * (0.5 * aaa * aaa) + (1 - x_bool) * (x_abs - 0.5), axis=(1, 2, 3))
    return lambda_rpn_reg * K.sum( (x_bool * (0.5 * aaa * aaa) + (1 - x_bool) * (x_abs - 0.5)), axis=(0, 1, 2, 3))


def rpn_loss_reg_quadratic(y_true, y_pred):
    """
    Leaky L1 norm loss, adaptated to exclude ignored pixels
    :param y_true: ground truth output [ndarray]
    :param y_pred: predicted output [ndarray]
    :return: loss function value []
    """

    mask = K.cast(K.not_equal(y_true, 0), 'float32')
    y_true=y_true*mask
    y_pred=y_pred*mask



    # x_abs = K.abs(aaa)
    # x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')

    return lambda_rpn_reg*mse(y_true,y_pred)

    # return lambda_rpn_reg * K.sum(x_bool * (0.5 * aaa * aaa) + (1 - x_bool) * (x_abs - 0.5), axis=(1, 2, 3))

def rpn_loss_cls(y_true, y_pred):
    """
    Categorical cross-entropy loss, adaptated to normalize by the number of valid samples (samples assigned to a class, not ignored)
    :param y_true: ground truth output [ndarray]
    :param y_pred: predicted output [ndarray]
    :return: loss function value []
    """

    # print(K.sum(binary_crossentropy(y_true, y_pred), axis=(0,1,2)) / K.sum(
    #     K.cast(K.greater(binary_crossentropy(y_true, y_pred), 0), 'float32'), axis=(0,1, 2)))

    return lambda_rpn_class * K.sum(binary_crossentropy(y_true, y_pred), axis=(0,1,2)) / K.sum(
        K.cast(K.greater(binary_crossentropy(y_true, y_pred), 0), 'float32'), axis=(0,1, 2))



import tensorflow as tf

def rpn_loss_cls_focal(y_true, y_pred):
    """
    Categorical cross-entropy loss, adaptated to normalize by the number of valid samples (samples assigned to a class, not ignored)
    :param y_true: ground truth output [ndarray]
    :param y_pred: predicted output [ndarray]
    :return: loss function value []
    """
    alpha=4
    gamma=2

    logprobs=tf.keras.metrics.categorical_crossentropy(y_true, y_pred)

    # print(K.sum(binary_crossentropy(y_true, y_pred), axis=(0,1,2)) / K.sum(
    #     K.cast(K.greater(binary_crossentropy(y_true, y_pred), 0), 'float32'), axis=(0,1, 2)))
    pt=tf.math.exp(logprobs) #??????
    return lambda_rpn_class * K.sum((alpha*((1-pt)**gamma))*logprobs, axis=(0,1,2)) / \
        K.sum( K.cast(K.greater(categorical_crossentropy(y_true, y_pred), 0), 'float32'), axis=(0,1, 2))


def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1

      where m = number of classes, c = class and o = observation

    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)

    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper

    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy

    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed


def rpn_loss_cls_new(y_true, y_pred):
    """
    Categorical cross-entropy loss, adaptated to normalize by the number of valid samples (samples assigned to a class, not ignored)
    :param y_true: ground truth output [ndarray]
    :param y_pred: predicted output [ndarray]
    :return: loss function value []
    """

    # print(K.sum(binary_crossentropy(y_true, y_pred), axis=(0,1,2)) / K.sum(
    #     K.cast(K.greater(binary_crossentropy(y_true, y_pred), 0), 'float32'), axis=(0,1, 2)))

    return lambda_rpn_class * K.sum(categorical_crossentropy(y_true, y_pred), axis=(0,1,2)) / K.sum(
        K.cast(K.greater(categorical_crossentropy(y_true, y_pred), 0), 'float32'), axis=(0,1, 2))

def rpn_loss_cls_freestyle(y_true, y_pred):
    """
    Categorical cross-entropy loss, adaptated to normalize by the number of valid samples (samples assigned to a class, not ignored)
    :param y_true: ground truth output [ndarray]
    :param y_pred: predicted output [ndarray]
    :return: loss function value []
    """

    validni = K.sum(y_true, axis=3)

    return lambda_rpn_class * K.sum(binary_crossentropy(y_true, y_pred), axis=(0,1, 2)) / K.sum(
        K.cast(K.greater(validni, 0), 'float32'), axis=(0,1, 2))



def binary_focal_loss_weighted(gamma=2., alpha=.25):
    """
    Binary form of focal loss.

      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)

      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.

    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)

    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # tf.print('y_true shape:', tf.shape(y_true))
        # print(y_true.shape)
        weights=y_true[:,:,:,1]
        y_pred=y_pred[:,:,:,0]
        # print(weights)
        # weights = tf.cast(weights, dtype=tf.float32)

        # print(f'WEights tensor shape {weights.shape}')
        y_true=y_true[:,:,:,0]
        # print(f'ytrue tensor shape {y_true.shape}')

        # tf.print('y_true shape:', tf.shape(y_true))
        # tf.print('y_pred shape:', tf.shape(y_pred))
        # tf.print('weights shape:', tf.shape(weights))

        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss= weight * cross_entropy
        # print(loss.shape)
        loss_weighted=weights*loss
        # Sum the losses in mini_batch
        loss_weighted= K.mean(K.sum(loss_weighted, axis=1))
        return loss_weighted

    return binary_focal_loss_fixed


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.

      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)

      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.

    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)

    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # tf.print('y_true shape:', tf.shape(y_true))
        # print(y_true.shape)
        # weights=y_true[:,:,:,1]
        # y_pred=y_pred[:,:,:,0]
        # print(weights)
        # weights = tf.cast(weights, dtype=tf.float32)

        # print(f'WEights tensor shape {weights.shape}')
        # y_true=y_true[:,:,:,0]
        # print(f'ytrue tensor shape {y_true.shape}')

        # tf.print('y_true shape:', tf.shape(y_true))
        # tf.print('y_pred shape:', tf.shape(y_pred))
        # tf.print('weights shape:', tf.shape(weights))

        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss= weight * cross_entropy
        # print(loss.shape)
        # loss_weighted=weights*loss
        # Sum the losses in mini_batch
        loss= K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed


def categorical_focal_loss_adapted(y_true, y_pred,alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1

      where m = number of classes, c = class and o = observation

    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)

    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper

    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy

    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)


    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred: A tensor resulting from a softmax
    :return: Output tensor.
    """

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    # Calculate Cross Entropy
    cross_entropy = -y_true * K.log(y_pred)

    # Calculate Focal Loss
    loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

    # Compute mean loss in mini_batch
    return K.mean(K.sum(loss, axis=-1))

    # return categorical_focal_loss_adapted

def binary_focal_loss_proba(y_true, y_pred,alpha,gamma):
    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred:  A tensor resulting from a sigmoid
    :return: Output tensor.
    """

    y_true = tf.cast(y_true, tf.float32)
    # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
    epsilon = K.epsilon()
    # Add the epsilon to prediction value
    # y_pred = y_pred + epsilon
    # Clip the prediciton value
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    # Calculate p_t
    p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
    # Calculate alpha_t
    alpha_factor = K.ones_like(y_true) * alpha
    alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    # Calculate cross entropy
    cross_entropy = -K.log(p_t)
    weight = alpha_t * K.pow((1 - p_t), gamma)
    # Calculate focal loss
    loss = weight * cross_entropy
    # Sum the losses in mini_batch
    loss = K.mean(K.sum(loss, axis=1))
    return loss


def binary_focal_loss_without_alpha(gamma=2.):
    """
    Binary form of focal loss.

      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)

      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.

    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)

    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        # alpha_factor = K.ones_like(y_true) * alpha
        # alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed

# def gloU_loss(y_true, y_pred):
#     y_true_classification = y_true[:, :num_classes]  # ? debug
#     y_true_regression = y_true[:, num_classes:]
#
#     indices = tf.where(tf.equal(y_true_classification, 1))
#     values_regressor = tf.gather_nd(y_true_regression, indices)



# num_classes=1
cls_coef=1
reg_coef_weight=0.01

# def giou_loss(y_true_cls,y_true_reg,y_pred_reg):
#     ones_indices = tf.where(tf.reduce_any(tf.equal(y_true_cls[:, :, :, :2], 1), axis=-1))
#     selected_values_reg_true = tf.gather_nd(y_true_reg, ones_indices)
#     selected_values_reg_pred = tf.gather_nd(y_pred_reg, ones_indices)
#
#     stride = tf.constant(8, dtype=tf.int64)
#     half_stride = tf.constant(4, dtype=tf.int64)
#
#     # Multiply all elements by the variable and add the constant
#     center_rows = ones_indices[:, 1] * stride + half_stride
#     center_cols = ones_indices[:, 2] * stride + half_stride
#
#     true_y = tf.cast(center_rows - (selected_values_reg_true[:, 2] / 2), dtype=tf.int64)
#     true_x = tf.cast(center_cols - (selected_values_reg_true[:, 3] / 2), dtype=tf.int64)
#     true_w = selected_values_reg_true[:, 3]
#     true_h = selected_values_reg_true[:, 2]
#     # max_rows_true=tf.cast(tf.add(min_rows_true, selected_values_reg_true[:,2]), dtype=tf.int64)
#     # max_cols_true=tf.cast(tf.add(min_cols_true, selected_values_reg_true[:,3]), dtype=tf.int64)
#
#     pred_y = tf.cast(center_rows - (selected_values_reg_pred[:, 2] / 2), dtype=tf.int64)
#     pred_x = tf.cast(center_cols - (selected_values_reg_pred[:, 3] / 2), dtype=tf.int64)
#     pred_w = selected_values_reg_pred[:, 3]
#     pred_h = selected_values_reg_pred[:, 2]
#     # max_rows_pred = tf.cast(tf.add(min_rows_pred, selected_values_reg_pred[:, 2]),dtype=tf.int64)
#     # max_cols_pred = tf.cast(tf.add(min_cols_pred, selected_values_reg_pred[:, 3]),dtype=tf.int64)
#
#     # Calculate intersection area
#     inter_w = tf.math.minimum(true_x + true_w, pred_x + pred_w) - tf.math.maximum(true_x, pred_x)
#     inter_h = tf.math.minimum(true_y + true_h, pred_y + pred_h) - tf.math.maximum(true_y, pred_y)
#     inter_area = tf.math.maximum(0.0, inter_w) * tf.math.maximum(0.0, inter_h)
#
#     # Calculate union area
#     true_area = true_w * true_h
#     pred_area = pred_w * pred_h
#     union_area = true_area + pred_area - inter_area
#
#     # Calculate GIOU
#     giou = inter_area / union_area
#     box_area = true_w * true_h + pred_w * pred_h
#
#     # Calculate C (smoothing term)
#     c = 1.0 - giou + (box_area - union_area) / box_area
#
#     # Calculate GIOU loss
#     giou_loss = 1.0 - giou + c
#
#     # Calculate bounding box area
#     # box_area = true_w * true_h + pred_w * pred_h
#
#     # Calculate C (smoothing term)
#     # c = 1.0 - giou
#
#     # Calculate GIOU loss
#     # giou_loss = 1.0 - giou + c
#
#     # Calculate mean loss over all bounding boxes
#     giou_loss = tf.reduce_mean(giou_loss)
#
#     return giou_loss


def iou_loss_new(y_true_cls, y_true_reg_r, y_pred_reg_r, coef_row, coef_col, coef_height, coef_width, img_height, img_width):
    # y_pred = tf.convert_to_tensor(y_pred)
    # if not y_pred.dtype.is_floating:
    #     y_pred = tf.cast(y_pred, tf.float32)
    # y_true = tf.cast(y_true, y_pred.dtype)
    # giou = tf.squeeze(_calculate_giou(y_pred, y_true, mode))



    y_true_cls = tf.cast(y_true_cls, dtype=tf.float32)
    y_true_reg_r = tf.cast(y_true_reg_r, dtype=tf.float32)
    y_pred_reg_r = tf.cast(y_pred_reg_r, dtype=tf.float32)
    constants = tf.constant([coef_row, coef_col, coef_height, coef_width], dtype=tf.float32)
    constants = tf.reshape(constants, (1, 1, 1, -1))
    # zero = tf.convert_to_tensor(0.0, dtype=tf.float32)
    # Multiply each element by the corresponding constant
    y_true_reg = y_true_reg_r * constants
    y_pred_reg = y_pred_reg_r * constants

    ones_indices = tf.where(tf.reduce_any(tf.equal(y_true_cls[:, :, :, :2], 1), axis=-1))

    if ones_indices.shape[0]==0:
        return 0
    selected_values_reg_true = tf.gather_nd(y_true_reg, ones_indices)
    selected_values_reg_pred = tf.gather_nd(y_pred_reg, ones_indices)

    stride = tf.constant(8., dtype=tf.float32)
    half_stride = tf.constant(4., dtype=tf.float32)

    # Multiply all elements by the variable and add the constant
    center_rows = tf.cast(ones_indices[:, 1], dtype=tf.float32) * stride + half_stride
    center_cols = tf.cast(ones_indices[:, 2], dtype=tf.float32) * stride + half_stride

    true_w_r = selected_values_reg_true[:, 3]
    true_h_r = selected_values_reg_true[:, 2]
    true_h = tf.multiply(true_h_r, img_height)
    true_w = tf.multiply(true_w_r, img_width)

    true_y0 = center_rows - (true_h / 2)
    true_y = tf.add(true_y0, selected_values_reg_true[:, 0])
    true_x0 = center_cols - (true_w / 2)
    true_x = tf.add(true_x0, selected_values_reg_true[:, 1])


    pred_w_r = selected_values_reg_pred[:, 3]
    pred_h_r = selected_values_reg_pred[:, 2]
    pred_h = tf.multiply(pred_h_r, img_height)
    pred_w = tf.multiply(pred_w_r, img_width)

    pred_y0 = center_rows - (pred_h / 2)
    pred_y = tf.add(pred_y0, selected_values_reg_pred[:, 0])
    pred_x0 = center_cols - (pred_w / 2)
    pred_x = tf.add(pred_x0, selected_values_reg_pred[:, 1])

    zero = tf.convert_to_tensor(0.0, pred_x.dtype)
    b1_ymin=true_y
    b1_xmin=true_x
    b1_ymax=true_y+true_h
    b1_xmax = true_x+true_w

    b2_ymin = pred_y
    b2_xmin = pred_x
    b2_ymax = pred_y + pred_h
    b2_xmax = pred_x + pred_w

    b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
    b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
    b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
    b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
    b1_area = b1_width * b1_height
    b2_area = b2_width * b2_height

    intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
    intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
    intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
    intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
    intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
    intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
    intersect_area = intersect_width * intersect_height

    union_area = b1_area + b2_area - intersect_area
    iou = tf.math.divide_no_nan(intersect_area, union_area)

    iou_loss=tf.reduce_mean(iou)
    return 1-iou_loss
def is_tensor_empty(tensor):
    return tf.equal(tf.size(tensor), 0)
def is_empty_case(*results):
    # Check if all results are -1 (use tf.reduce_all for TensorFlow operations)
    return tf.reduce_all([tf.reduce_all(tf.equal(r, -1.0)) for r in results])
def form_bboxes(y_true_cls, y_true_reg_r, y_pred_reg_r, coef_row, coef_col, coef_height, coef_width, img_height, img_width):
    '''

    :param y_true_cls: ground truth matrices for classification (b_size,rows,cols,depth); depth=num_of_classes
    :param y_true_reg_r: ground truth matrices for classification (b_size,rows,cols,depth); depth=4 (delta_r,delta_col, h, w)
    :param y_pred_reg_r: predicted matrices for regression (b_size,rows,cols,depth); depth=4 (delta_r,delta_col, h, w)
    :param coef_row: maximum of matrices with depth 0 from the whole dataset.. normalized before training
    :param coef_col:
    :param coef_height:
    :param coef_width:
    :param img_height:
    :param img_width:
    :return: 2 tensors with shape(num_bboxes, 4).. 4 stands for [ymin,xmin,ymax,xmax]. Tensor 1 for ground truth bounding boxes and Tensor 2 for predicted bound boxes
    '''
    y_true_cls = tf.cast(y_true_cls, dtype=tf.float32)
    y_true_reg_r = tf.cast(y_true_reg_r, dtype=tf.float32)
    y_pred_reg_r = tf.cast(y_pred_reg_r, dtype=tf.float32)
    constants = tf.constant([coef_row, coef_col, coef_height, coef_width], dtype=tf.float32)
    constants = tf.reshape(constants, (1, 1, 1, -1))
    # zero = tf.convert_to_tensor(0.0, dtype=tf.float32)
    # Multiply each element by the corresponding constant
    y_true_reg = y_true_reg_r * constants
    y_pred_reg = y_pred_reg_r * constants

    ones_indices = tf.where(tf.reduce_any(tf.equal(y_true_cls[:, :, :, :2], 1), axis=-1))

    if is_tensor_empty(ones_indices):
        tf.print("nema centri",is_tensor_empty(ones_indices))
        return (
            tf.constant(-1.0),
            tf.constant(-1.0),
            tf.constant(-1.0),
            tf.constant(-1.0),
            tf.constant(-1.0),
            tf.constant(-1.0),
            tf.constant(-1.0),
            tf.constant(-1.0),
        )
    selected_values_reg_true = tf.gather_nd(y_true_reg, ones_indices)
    selected_values_reg_pred = tf.gather_nd(y_pred_reg, ones_indices)

    stride = tf.constant(8., dtype=tf.float32)
    half_stride = tf.constant(4., dtype=tf.float32)

    # Multiply all elements by the variable and add the constant
    center_rows = tf.cast(ones_indices[:, 1], dtype=tf.float32) * stride + half_stride
    center_cols = tf.cast(ones_indices[:, 2], dtype=tf.float32) * stride + half_stride

    true_w_r = selected_values_reg_true[:, 3]
    true_h_r = selected_values_reg_true[:, 2]
    true_h = tf.multiply(true_h_r, img_height)
    true_w = tf.multiply(true_w_r, img_width)

    true_y0 = center_rows - (true_h / 2)
    true_y = tf.add(true_y0, selected_values_reg_true[:, 0])
    true_x0 = center_cols - (true_w / 2)
    true_x = tf.add(true_x0, selected_values_reg_true[:, 1])

    pred_w_r = selected_values_reg_pred[:, 3]
    pred_h_r = selected_values_reg_pred[:, 2]
    pred_h = tf.multiply(pred_h_r, img_height)
    pred_w = tf.multiply(pred_w_r, img_width)

    pred_y0 = center_rows - (pred_h / 2)
    pred_y = tf.add(pred_y0, selected_values_reg_pred[:, 0])
    pred_x0 = center_cols - (pred_w / 2)
    pred_x = tf.add(pred_x0, selected_values_reg_pred[:, 1])

    # zero = tf.convert_to_tensor(0.0, pred_x.dtype)
    b1_ymin = true_y
    b1_xmin = true_x
    b1_ymax = true_y + true_h
    b1_xmax = true_x + true_w

    b2_ymin = pred_y
    b2_xmin = pred_x
    b2_ymax = pred_y + pred_h
    b2_xmax = pred_x + pred_w
    return b1_ymin,b1_xmin,b1_ymax,b1_xmax, b2_ymin,b2_xmin,b2_ymax,b2_xmax


def diou_loss(y_true_cls, y_true_reg_r, y_pred_reg_r, coef_row, coef_col, coef_height, coef_width, img_height, img_width):

    b1_ymin, b1_xmin, b1_ymax, b1_xmax, b2_ymin, b2_xmin, b2_ymax, b2_xmax=form_bboxes(y_true_cls, y_true_reg_r, y_pred_reg_r, coef_row, coef_col, coef_height, coef_width, img_height, img_width)
    if is_empty_case(b1_ymin, b1_xmin, b1_ymax, b1_xmax, b2_ymin, b2_xmin, b2_ymax, b2_xmax):
        return tf.constant(0.0)

    # tf.print("bbox", b1_ymin, b1_xmin, b1_ymax, b1_xmax, b2_ymin, b2_xmin, b2_ymax, b2_xmax)
    zero = tf.convert_to_tensor(0.0, b1_ymin.dtype)

    b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
    b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
    b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
    b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
    b1_area = b1_width * b1_height
    b2_area = b2_width * b2_height

    intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
    intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
    intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
    intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
    intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
    intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
    intersect_area = intersect_width * intersect_height

    union_area = b1_area + b2_area - intersect_area
    iou = tf.math.divide_no_nan(intersect_area, union_area)
    # tf.print("iou",iou)
    centerx_b1=(b1_xmin+b1_xmax)/2
    centery_b1=(b1_ymin+b1_ymax)/2

    centerx_b2 = (b2_xmin + b2_xmax) / 2
    centery_b2 = (b2_ymin + b2_ymax) / 2

    c_b1 = tf.concat([centerx_b1, centery_b1], axis=-1)
    c_b2 = tf.concat([centerx_b2, centery_b2], axis=-1)

    distances = tf.norm(c_b1 - c_b2, axis=-1)

    enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
    enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
    enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
    enclose_xmax = tf.maximum(b1_xmax, b2_xmax)

    e_points_min = tf.concat([enclose_xmin, enclose_ymin], axis=-1)
    e_points_max = tf.concat([enclose_xmax, enclose_ymax], axis=-1)

    c_dist = tf.norm(e_points_min - e_points_max, axis=-1)

    distances_squared=tf.square(distances)
    c_dist_squared=tf.square(c_dist)

    diou_l=1-iou+tf.math.divide_no_nan(distances_squared,c_dist_squared)

    diou_loss=tf.math.reduce_mean(diou_l)
    return diou_loss

#note da se smeni
def ciou_loss(y_true_cls, y_true_reg_r, y_pred_reg_r, coef_row, coef_col, coef_height, coef_width, img_height, img_width):

    b1_ymin, b1_xmin, b1_ymax, b1_xmax, b2_ymin, b2_xmin, b2_ymax, b2_xmax=form_bboxes(y_true_cls, y_true_reg_r, y_pred_reg_r, coef_row, coef_col, coef_height, coef_width, img_height, img_width)


    zero = tf.convert_to_tensor(0.0, b1_ymin.dtype)

    b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
    b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
    b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
    b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
    b1_area = b1_width * b1_height
    b2_area = b2_width * b2_height

    intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
    intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
    intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
    intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
    intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
    intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
    intersect_area = intersect_width * intersect_height

    union_area = b1_area + b2_area - intersect_area
    iou = tf.math.divide_no_nan(intersect_area, union_area)

    centerx_b1=(b1_xmin+b1_xmax)/2
    centery_b1=(b1_ymin+b1_ymax)/2

    centerx_b2 = (b2_xmin + b2_xmax) / 2
    centery_b2 = (b2_ymin + b2_ymax) / 2

    c_b1 = tf.concat([centerx_b1, centery_b1], axis=-1)
    c_b2 = tf.concat([centerx_b2, centery_b2], axis=-1)

    distances = tf.norm(c_b1 - c_b2, axis=-1)

    enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
    enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
    enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
    enclose_xmax = tf.maximum(b1_xmax, b2_xmax)

    e_points_min = tf.concat([enclose_xmin, enclose_ymin], axis=-1)
    e_points_max = tf.concat([enclose_xmax, enclose_ymax], axis=-1)

    c_dist = tf.norm(e_points_min - e_points_max, axis=-1)

    distances_squared=tf.square(distances)
    c_dist_squared=tf.square(c_dist)

    diou_l=1-iou+tf.math.divide_no_nan(distances_squared,c_dist_squared)

    diou_loss=tf.math.reduce_mean(diou_l)
    return diou_loss
def giou_loss(y_true_cls, y_true_reg_r, y_pred_reg_r, coef_row, coef_col, coef_height, coef_width, img_height, img_width):
    # y_true_reg[:, :, :, 0] = y_true_reg[:, :, :, 0] * coef_row
    # y_true_reg[:, :, :, 1] = y_true_reg[:, :, :, 1] * coef_col
    # y_true_reg[:, :, :, 2] = y_true_reg[:, :, :, 2] * coef_height
    # y_true_reg[:, :, :, 3] = y_true_reg[:, :, :, 3] * coef_width
    #
    # y_pred_reg[:, :, :, 0] = y_pred_reg[:, :, :, 0] * coef_row
    # y_pred_reg[:, :, :, 1] = y_pred_reg[:, :, :, 1] * coef_col
    # y_pred_reg[:, :, :, 2] = y_pred_reg[:, :, :, 2] * coef_height
    # y_pred_reg[:, :, :, 3] = y_pred_reg[:, :, :, 3] * coef_width

    y_true_cls = tf.cast(y_true_cls, dtype=tf.float32)
    y_true_reg_r = tf.cast(y_true_reg_r, dtype=tf.float32)
    y_pred_reg_r = tf.cast(y_pred_reg_r, dtype=tf.float32)
    constants = tf.constant([coef_row, coef_col, coef_height, coef_width], dtype=tf.float32)
    constants = tf.reshape(constants, (1, 1, 1, -1))
    zero = tf.convert_to_tensor(0.0, dtype=tf.float32)
    # Multiply each element by the corresponding constant
    y_true_reg = y_true_reg_r * constants
    y_pred_reg = y_pred_reg_r * constants

    ones_indices = tf.where(tf.reduce_any(tf.equal(y_true_cls[:, :, :, :2], 1), axis=-1))

    if ones_indices.shape[0] == 0:
        return 0
    selected_values_reg_true = tf.gather_nd(y_true_reg, ones_indices)
    selected_values_reg_pred = tf.gather_nd(y_pred_reg, ones_indices)

    stride = tf.constant(8., dtype=tf.float32)
    half_stride = tf.constant(4., dtype=tf.float32)

    # Multiply all elements by the variable and add the constant
    center_rows = tf.cast(ones_indices[:, 1], dtype=tf.float32) * stride + half_stride
    center_cols = tf.cast(ones_indices[:, 2], dtype=tf.float32) * stride + half_stride

    true_w_r = selected_values_reg_true[:, 3]
    true_h_r = selected_values_reg_true[:, 2]
    true_h = tf.multiply(true_h_r, img_height)
    true_w = tf.multiply(true_w_r, img_width)

    true_y0 = center_rows - (true_h / 2)
    true_y = tf.add(true_y0, selected_values_reg_true[:, 0])
    true_x0 = center_cols - (true_w / 2)
    true_x = tf.add(true_x0, selected_values_reg_true[:, 1])

    pred_w_r = selected_values_reg_pred[:, 3]
    pred_h_r = selected_values_reg_pred[:, 2]
    pred_h = tf.multiply(pred_h_r, img_height)
    pred_w = tf.multiply(pred_w_r, img_width)

    pred_y0 = center_rows - (pred_h / 2)
    pred_y = tf.add(pred_y0, selected_values_reg_pred[:, 0])
    pred_x0 = center_cols - (pred_w / 2)
    pred_x = tf.add(pred_x0, selected_values_reg_pred[:, 1])

    zero = tf.convert_to_tensor(0.0, pred_x.dtype)
    b1_ymin = true_y
    b1_xmin = true_x
    b1_ymax = true_y + true_h
    b1_xmax = true_x + true_w

    b2_ymin = pred_y
    b2_xmin = pred_x
    b2_ymax = pred_y + pred_h
    b2_xmax = pred_x + pred_w

    b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
    b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
    b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
    b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
    b1_area = b1_width * b1_height
    b2_area = b2_width * b2_height

    intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
    intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
    intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
    intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
    intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
    intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
    intersect_area = intersect_width * intersect_height

    union_area = b1_area + b2_area - intersect_area
    iou = tf.math.divide_no_nan(intersect_area, union_area)

    enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
    enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
    enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
    enclose_xmax = tf.maximum(b1_xmax, b2_xmax)
    enclose_width = tf.maximum(zero, enclose_xmax - enclose_xmin)
    enclose_height = tf.maximum(zero, enclose_ymax - enclose_ymin)
    enclose_area = enclose_width * enclose_height

    # giou_loss=1-iou
    giou=iou-tf.math.divide_no_nan((enclose_area - union_area), enclose_area)
    giou_loss=1-giou

    #
    # box_area = true_w * true_h + pred_w * pred_h
    #
    # # Calculate C (smoothing term)
    # c = 1.0 - giou + (box_area - union_area) / box_area
    #
    # # Calculate GIOU loss
    # giou_loss = 1.0 - giou + c

    # Calculate bounding box area
    # box_area = true_w * true_h + pred_w * pred_h

    # Calculate C (smoothing term)
    # c = 1.0 - giou

    # Calculate GIOU loss
    # giou_loss = 1.0 - giou + c

    # Calculate mean loss over all bounding boxes
    giou_loss = tf.reduce_mean(giou_loss)

    return giou_loss
def combined_loss(coef_row, coef_col, coef_height, coef_width, img_height, img_width,num_classes):

    def combined_loss_fixed(y_true, y_pred):


        # with tf.Session() as sess:
        #     y_tr_n=sess.run(y_true)
        # print(y_tr_n.shape)
        y_true_classification = y_true[:,:,:, :num_classes] #? debug
        y_pred_classification = y_pred[:,:,:, :num_classes] #? debug
        y_true_regression = y_true[:,:,:, num_classes:]
        y_pred_regression = y_pred[:,:,:, num_classes:]


        # Assuming you have focal loss for classification and GloU for regression
        classification_loss = categorical_focal_loss_adapted(y_true_classification, y_pred_classification,[1,1,1,1],2.) #?
        # tf.print("cls loss:",classification_loss)
        regression_loss = diou_loss(y_true_classification,y_true_regression,y_pred_regression,coef_row, coef_col, coef_height, coef_width, img_height, img_width)
        # tf.print("reg loss:",regression_loss)

        # You can choose appropriate weights for the two losses
        total_loss = cls_coef * classification_loss+reg_coef_weight*regression_loss
        #total_loss = regression_loss
        # tf.print("total loss", total_loss)
        return total_loss

    return combined_loss_fixed



class LossLogger(Callback):
    def __init__(self,  num_classes,img_height, img_width,filename, validation_generator=None):
        self.filename=filename
        self.num_classes=num_classes
        self.img_height= img_height
        self.img_width= img_width
        self.validation_generator=validation_generator

    def on_epoch_end(self, epoch, logs=None):
        # Get the current model
        model = self.model

        # Get validation data using the generator
        val_data = self.validation_generator.__getitem__(index=0)

        # Predict on validation data
        y_pred = model.predict(val_data[0])

        # Extract true labels
        y_true = val_data[1]

        # Compute classification loss and regression loss
        y_true_classification = y_true[:, :, :, :self.num_classes]
        y_pred_classification = y_pred[:, :, :, :self.num_classes]
        y_true_regression = y_true[:, :, :, self.num_classes:]
        y_pred_regression = y_pred[:, :, :, self.num_classes:]

        # classification_loss = binary_focal_loss_proba(y_true_classification, y_pred_classification, [1, 1, 1, 1], 2.)
        classification_loss = helper_losses.binary_focal_loss_proba(y_true_classification, y_pred_classification, [1, 1, 1, 1], 2.)
        regression_loss = diou_loss(y_true_classification, y_true_regression, y_pred_regression, 1,1,1,1, self.img_height, self.img_width)
        cls_loss=classification_loss.numpy()
        reg_loss=regression_loss.numpy()

        # with tf.Session() as sess:
        #     cls_loss, reg_loss = sess.run([classification_loss, regression_loss ])

        # Save the losses to a file
        with open(self.filename, 'a') as file:
            file.write(f'Epoch {epoch + 1} - Classification Loss: {cls_loss}, Regression Loss: {reg_loss}\n')
