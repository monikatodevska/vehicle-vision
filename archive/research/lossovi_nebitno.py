import tensorflow as tf
from keras.objectives import categorical_crossentropy, binary_crossentropy

tf.keras.losses.BinaryCrossentropy(
    from_logits=False,
    label_smoothing=0.0,
    reduction="auto",
    name="binary_crossentropy",
)

# y_true = [[0.0, 1.0]]
# y_pred = [[0.99, 0.01]]
# bce = categorical_crossentropy(y_true, y_pred)
import numpy as np
# y_true =[0.0, 0.0]
# y_pred = [0.99, 0.01]


y_true= [[0, 1],
      [1, 0],
      [1, 0]]

y_pred=[[0.2, 0.8],
            [0.7, 0.3],
            [0.5, 0.5]]

# loss = tf.keras.losses.CategoricalCrossentropy()
bce = categorical_crossentropy(y_true, y_pred)

# loss_temp=loss(np.asarray(y_true), np.asarray(y_pred))
# loss_np=loss_temp.eval(session=tf.compat.v1.Session())
# print(loss_np)
# print(bce)

a = bce.eval(session=tf.compat.v1.Session())
print(a)
print(type(a))



