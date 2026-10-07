import tensorflow as tf
import keras.backend as K
import numpy as np
import os
from tensorflow.python.ops import math_ops
from keras.layers import Conv2D, MaxPool2D, Input, concatenate,Cropping2D
from keras.models import Model, model_from_json
from tensorflow.python.framework import ops
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

lambda_rpn_reg = 1


def huber_loss(y_true, y_pred, delta=1.0):
  """Computes Huber loss value.

  For each value x in `error=y_true-y_pred`, the following is calculated:

  ```
  0.5 * x^2                  if |x| <= d
  0.5 * d^2 + d * (|x| - d)  if |x| > d
  ```
  where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss

  Args:
    y_true: tensor of true targets.
    y_pred: tensor of predicted targets.
    delta: A float, the point where the Huber loss function changes from a
      quadratic to linear.

  Returns:
    Tensor with one scalar loss entry per sample.
  """
  y_pred = math_ops.cast(y_pred, dtype=K.floatx())
  y_true = math_ops.cast(y_true, dtype=K.floatx())
  error = math_ops.subtract(y_pred, y_true)
  abs_error = math_ops.abs(error)
  quadratic = math_ops.minimum(abs_error, delta)
  linear = math_ops.subtract(abs_error, quadratic)
  final=math_ops.add(
      math_ops.multiply(
          ops.convert_to_tensor(0.5, dtype=quadratic.dtype),
          math_ops.multiply(quadratic, quadratic)),
      math_ops.multiply(delta, linear))

  b = final.eval(session=tf.compat.v1.Session())

  return math_ops.add(
      math_ops.multiply(
          ops.convert_to_tensor(0.5, dtype=quadratic.dtype),
          math_ops.multiply(quadratic, quadratic)),
      math_ops.multiply(delta, linear))


def rpn_loss_reg(y_true, y_pred):
    """
    Leaky L1 norm loss, adaptated to exclude ignored pixels
    :param y_true: ground truth output [ndarray]
    :param y_pred: predicted output [ndarray]
    :return: loss function value []
    """

    mask = K.cast(K.not_equal(y_true, 0), 'float32')

    aaa = y_true - y_pred
    aaa = aaa * mask

    x_abs = K.abs(aaa)
    x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')

    # return lambda_rpn_reg * K.sum(x_bool * (0.5 * aaa * aaa) + (1 - x_bool) * (x_abs - 0.5), axis=(1, 2, 3))
    return lambda_rpn_reg * K.sum((x_bool * (0.5 * aaa * aaa) + (1 - x_bool) * (x_abs - 0.5)), axis=(0, 1, 2, 3))
    # return lambda_rpn_reg * K.sum((x_bool * (0.5 * aaa * aaa) + (1 - x_bool) * (x_abs - 0.5)), axis=(1, 2, 3))
    # return lambda_rpn_reg * (K.sum((x_bool * (0.5 * aaa * aaa) + (1 - x_bool) * (x_abs - 0.5)), axis=(0, 1, 2, 3)) / y_true.shape[-1]) / y_true.shape[0]
    # return lambda_rpn_reg * (K.sum((x_bool * (0.5 * aaa * aaa) + (1 - x_bool) * (x_abs - 0.5)), axis=(0,)) / y_true.shape[-1]) / y_true.shape[0]


def construct_model(input_shape):
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

    x_reg_2 = Conv2D(filters=2, kernel_size=(1, 1), padding='same', activation='linear', kernel_initializer='zeros', name="out_reg")(f1)

    model = Model(input_layer, x_reg_2)

    return model


#########################################



y_true = np.zeros(shape=(3, 1, 1, 4))
y_pred = np.zeros(shape=(3, 1, 1, 4))

y_true[0, 0, 0, 1] = 1
y_pred[0, 0, 0, 1] = 0.8

y_true[1, 0, 0, 1] = 1
y_pred[1, 0, 0, 1] = 0.3

y_true[1, 0, 0, 0] = 1
y_pred[1, 0, 0, 0] = 0.5

print(y_true.shape)

# Using default 'auto'/'sum_over_batch_size' reduction type.
loss_val = rpn_loss_reg(y_true, y_pred)
a = loss_val.eval(session=tf.compat.v1.Session())
print(a)


h1 = tf.keras.losses.Huber()
h1_val = h1(y_true, y_pred)
b1 = h1_val.eval(session=tf.compat.v1.Session())
print(b1)


h=huber_loss(y_true,y_pred)

b = h.eval(session=tf.compat.v1.Session())
print(b)


print(a)
print(type(a))

print(b)
print(type(b))



#########################################3

lr = 0.1

model = construct_model(input_shape=(3, 3, 1))  # build model architecture

# compile model
model.compile(loss={'out_reg': rpn_loss_reg},
              optimizer=Adam(lr=lr),
              metrics=['accuracy'])

# --- fit model ---
model_checkpoint = ModelCheckpoint(filepath=os.path.join(modelsPath, 'checkpoint-{epoch:03d}-{loss:.4f}.hdf5'),
                                   # epoch number and val accuracy will be part of the weight file name
                                   monitor='loss',  # metric to monitor when selecting weight checkpoints to save
                                   verbose=1,
                                   save_best_only=True)  # True saves only the weights after epochs where the monitored value (val accuracy) is improved



##########################################

def wrapper(param1):
    def custom_loss_1(y_true, y_pred):
      diff = math_ops.squared_difference(y_pred, y_true)  #squared difference
      loss = K.mean(diff, axis=-1) #mean

      loss_num = loss.eval(session=tf.compat.v1.Session())

      loss = loss / param1
      return loss
    return custom_loss_1

loss = wrapper(1.0)

final_loss = loss(y_true=[[10.0,7.0], [2.0,3.0]], y_pred=[[8.0, 6.0], [3.0, 4.0]])
final_loss_num = final_loss.eval(session=tf.compat.v1.Session())
print(f"Final Loss is {final_loss_num}")