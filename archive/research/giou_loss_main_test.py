import tensorflow as tf
import pickle
import numpy as np
im_true=r'C:\Users\User\Desktop\test\image030884.bmp'
gt_true=r'C:\Users\User\Desktop\test\image030884.txt'


im_pred=r'C:\Users\User\Desktop\test\image030890.bmp'
gt_pred=r'C:\Users\User\Desktop\test\image030890.txt'

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

    y_true_cls=tf.cast(y_true_cls, dtype=tf.float32)
    y_true_reg_r=tf.cast(y_true_reg_r, dtype=tf.float32)
    y_pred_reg_r=tf.cast(y_pred_reg_r, dtype=tf.float32)
    constants = tf.constant([coef_row, coef_col, coef_height, coef_width], dtype=tf.float32)
    constants = tf.reshape(constants, (1, 1, 1, -1))
    zero= tf.convert_to_tensor(0.0, dtype=tf.float32)
    # Multiply each element by the corresponding constant
    #vrati za trening!
    y_true_reg = y_true_reg_r
    y_pred_reg = y_pred_reg_r


    with tf.Session() as sess:
        # Evaluate the tensors and convert them to NumPy arrays
        numpy_array1, numpy_array2 = sess.run([y_true_reg[0,:,:,0], y_pred_reg])

    print(numpy_array1)
    # print(numpy_array2)

    ones_indices = tf.where(tf.reduce_any(tf.equal(y_true_cls[:, :, :, :2], 1), axis=-1))
    selected_values_reg_true = tf.gather_nd(y_true_reg, ones_indices)
    selected_values_reg_pred = tf.gather_nd(y_pred_reg, ones_indices)



    stride = tf.constant(8., dtype=tf.float32)
    half_stride = tf.constant(4., dtype=tf.float32)

    # Multiply all elements by the variable and add the constant
    center_rows = tf.cast(ones_indices[:, 1] , dtype=tf.float32)* stride + half_stride
    center_cols = tf.cast(ones_indices[:, 2], dtype=tf.float32) * stride + half_stride

    with tf.Session() as sess:
        # Evaluate the tensors and convert them to NumPy arrays
        numpy_array1, numpy_array2, numpy_array3 = sess.run([selected_values_reg_true,center_rows,center_cols])

    # print(numpy_array1)
    # print(numpy_array2)
    # print(numpy_array3)



    true_w_r = selected_values_reg_true[:, 3]
    true_h_r = selected_values_reg_true[:, 2]
    true_h = tf.multiply(true_h_r, img_height)
    true_w = tf.multiply(true_w_r, img_width)

    true_y0 = center_rows - (true_h / 2)
    true_y = tf.add(true_y0, selected_values_reg_true[:, 0])
    true_x0 = center_cols - (true_w / 2)
    true_x = tf.add(true_x0, selected_values_reg_true[:, 1])

    # --------------
    # h = h_percent * img_dims[0]
    # w = w_percent * img_dims[1]
    # print(h)
    # print(w)
    # bbox top left point
    # min_row = np.int(center_row - np.round(h / 2))
    # min_col = np.int(center_col - np.round(w / 2))

    # adjust position and size with regressor predictions
    #     min_row_adj = np.int(min_row + delta_r)
    #     min_col_adj = np.int(min_col + delta_c)
    #     tf.multiply(tensor_a, tensor_b)
    #
    # -------
    # max_rows_true=tf.cast(tf.add(min_rows_true, selected_values_reg_true[:,2]), dtype=tf.int64)
    # max_cols_true=tf.cast(tf.add(min_cols_true, selected_values_reg_true[:,3]), dtype=tf.int64)
    pred_w_r = selected_values_reg_pred[:, 3]
    pred_h_r = selected_values_reg_pred[:, 2]
    pred_h = tf.multiply(pred_h_r, img_height)
    pred_w = tf.multiply(pred_w_r, img_width)

    pred_y0 = center_rows - (pred_h / 2)
    pred_y = tf.add(pred_y0, selected_values_reg_pred[:, 0])
    pred_x0 = center_cols - (pred_w / 2)
    pred_x = tf.add(pred_x0, selected_values_reg_pred[:, 1])


    # max_rows_pred = tf.cast(tf.add(min_rows_pred, selected_values_reg_pred[:, 2]),dtype=tf.int64)
    # max_cols_pred = tf.cast(tf.add(min_cols_pred, selected_values_reg_pred[:, 3]),dtype=tf.int64)

    # Calculate intersection area
    #dobro

    # return true_x,true_y,true_w,true_h,pred_x,pred_y,pred_w,pred_h

    inter_w = tf.math.minimum(true_x + true_w, pred_x + pred_w) - tf.math.maximum(true_x, pred_x)
    inter_h = tf.math.minimum(true_y + true_h, pred_y + pred_h) - tf.math.maximum(true_y, pred_y)
    inter_area = tf.math.maximum(0.0, inter_w) * tf.math.maximum(0.0, inter_h)
    #
    #
    #
    # # Calculate union area
    # #dobro e
    true_area = true_w * true_h
    pred_area = pred_w * pred_h
    union_area = true_area + pred_area - inter_area

    # Calculate GIOU
    iou = tf.math.divide_no_nan(inter_area, union_area)

    return iou
    # e_ymin=tf.minimum(true_y, pred_y)
    # e_xmin=tf.minimum(true_x, pred_x)
    # e_ymax=tf.maximum(true_y+true_h, pred_y + pred_h)
    # e_xmax=tf.maximum(true_x + true_w, pred_x + pred_w)
    # e_height=tf.maximum(zero,e_ymax-e_ymin)
    # e_width=tf.maximum(zero,e_xmax-e_xmin)
    # enclose_area=e_height*e_width
    #
    # giou=iou-tf.math.divide_no_nan((enclose_area - union_area), enclose_area)
    # giou_loss=1-giou
    # # return giou_lossloss
    # #
    # # #
    # # # box_area = true_w * true_h + pred_w * pred_h
    # # #
    # # # # Calculate C (smoothing term)
    # # # c = 1.0 - giou + (box_area - union_area) / box_area
    # # #
    # # # # Calculate GIOU loss
    # # # giou_loss = 1.0 - giou + c
    # #
    # # # Calculate bounding box area
    # # # box_area = true_w * true_h + pred_w * pred_h
    # #
    # # # Calculate C (smoothing term)
    # # # c = 1.0 - giou
    # #
    # # # Calculate GIOU loss
    # # # giou_loss = 1.0 - giou + c
    # #
    # # Calculate mean loss over all bounding boxes
    # giou_loss = tf.reduce_mean(giou_loss)
    #
    # return giou_loss

#
# # Generate a tensor with shape (2, 3, 3, 4) containing random ones and zeros
# y_true_cls = tf.random.uniform(shape=(2, 3, 3, 4), minval=0, maxval=2, dtype=tf.int64)
#
# # Generate a tensor with shape (2, 3, 3, 4) containing random integers
# y_true_reg_r = tf.random.uniform(shape=(2, 3, 3, 4), minval=0, maxval=10, dtype=tf.int32)
#
#
# # Generate a tensor with shape (2, 3, 3, 4) containing random integers
# y_pred_reg_r = tf.random.uniform(shape=(2, 3, 3, 4), minval=0, maxval=10, dtype=tf.int32)
#
# constants = tf.constant([1.1, 2.0, 3.1, 4.1], dtype=tf.float32)
#
# # Perform element-wise division on each depth matrix
# y_true_reg_r = tf.cast(y_true_reg_r, dtype=tf.float32) / constants
# y_pred_reg_r = tf.cast(y_pred_reg_r, dtype=tf.float32) / constants
#
# with tf.Session() as sess:
#     # Run the operations
#     cls_tensor_value, cls1_tensor_value, divided_tensor_value = sess.run([y_true_cls[0,:,:,0],y_true_cls[0,:,:,1], y_true_reg_r[0,:,:,0]])
#
#     # Print the tensors
#     print("Original Integer Tensor:")
#     print(cls_tensor_value)
#
#     print("\nConstants:")
#     print(cls1_tensor_value)
#
#     print("\nDivided Tensor:")
#     print(divided_tensor_value)
from copy import deepcopy
fid1=open(gt_true, 'rb')

out_class_dims=pickle.load(fid1)
out_class_back=pickle.load(fid1)
out_reg_dims=pickle.load(fid1)
out_reg_back=pickle.load(fid1)
fid1.close()
out_class_true= np.reshape(out_class_back, (out_class_dims[0], out_class_dims[1], out_class_dims[2]))
out_reg_true = np.reshape(out_reg_back, (out_reg_dims[0], out_reg_dims[1], out_reg_dims[2]))




fid2=open(gt_pred, 'rb')

out_class_dims=pickle.load(fid2)
out_class_back=pickle.load(fid2)
out_reg_dims=pickle.load(fid2)
out_reg_back=pickle.load(fid2)
fid2.close()
out_class_true2=np.reshape(out_class_back, (out_class_dims[0], out_class_dims[1], out_class_dims[2]))
out_reg_true2 = np.reshape(out_reg_back, (out_reg_dims[0], out_reg_dims[1], out_reg_dims[2]))


out_class_pred1=deepcopy(out_class_true)
out_reg_pred1=deepcopy(out_reg_true)

out_class_pred2=deepcopy(out_class_true2)
out_reg_pred2=deepcopy(out_reg_true2)

out_class_true=np.expand_dims(out_class_true, axis=0)
out_class_true2=np.expand_dims(out_class_true2, axis=0)

out_class_pred1=np.expand_dims(out_class_pred1, axis=0)
out_class_pred2=np.expand_dims(out_class_pred2, axis=0)

out_reg_true=np.expand_dims(out_reg_true, axis=0)
out_reg_true2=np.expand_dims(out_reg_true2, axis=0)


out_reg_pred1=np.expand_dims(out_reg_pred1, axis=0)
out_reg_pred2=np.expand_dims(out_reg_pred2, axis=0)






out_class_concat_true=np.concatenate((out_class_true,out_class_true2),axis=0)
out_reg_concat_true=np.concatenate((out_reg_true,out_reg_true2),axis=0)
out_class_concat_pred=np.concatenate((out_class_pred1,out_class_pred2),axis=0)
out_reg_concat_pred=np.concatenate((out_reg_pred1,out_reg_pred2),axis=0)



#
#
# out_class_true=np.expand_dims(out_class_true, axis=0)
# out_reg_pred=np.expand_dims(out_reg_pred, axis=0)
# out_reg_true=np.expand_dims(out_reg_true, axis=0)


out_class_true_t=tf.convert_to_tensor(out_class_concat_true)
out_reg_pred_t=tf.convert_to_tensor(out_reg_concat_pred)
out_reg_true_t=tf.convert_to_tensor(out_reg_concat_true)

np.set_printoptions(threshold=np.inf)

print(out_reg_true[0,:,:,0].shape)
print(out_reg_true[0,:,:,0])
#101.0,98.0,0.5982404692082112,0.376953125
# true_x,true_y,true_w,true_h,pred_x,pred_y,pred_w,pred_h=giou_loss(out_class_true_t,out_reg_true_t,out_reg_pred_t,1,1,1,1,341,512)
# true_x, true_y, true_w,true_h, pred_x,pred_y, pred_w, pred_h=giou_loss(out_class_true_t,out_reg_true_t,out_reg_pred_t,1,1,1,1,341,512)
iou=giou_loss(out_class_true_t,out_reg_true_t,out_reg_pred_t,1,1,1,1,341,512)



import cv2

im=cv2.imread(im_true,0)
im_t2=cv2.imread(im_pred,0)


with tf.Session() as sess:
    # Evaluate the tensors and convert them to NumPy arrays
    iou_l= sess.run([iou])

print(iou_l)
# with tf.Session() as sess:
#     # Evaluate the tensors and convert them to NumPy arrays
#     true_x, true_y, true_w, true_h,pred_x,pred_y, pred_w, pred_h = sess.run([true_x, true_y, true_w,true_h, pred_x,pred_y, pred_w, pred_h])



# for ind in range(len(true_y)):
#     cv2.rectangle(im, (int(true_x[ind]),int(true_y[ind])),(int(true_x[ind]+true_w[ind]), int(true_y[ind]+true_h[ind])), color=(255,0,0))
# cv2.imshow('sl',im)
# cv2.waitKey(0)
#
# for ind in range(len(true_y)):
#     cv2.rectangle(im1, (int(pred_x[ind]),int(pred_y[ind])),(int(pred_x[ind]+pred_w[ind]), int(pred_y[ind]+pred_h[ind])), color=(255,0,0))
# cv2.imshow('sl1',im1)
# cv2.waitKey(0)
# print(1)