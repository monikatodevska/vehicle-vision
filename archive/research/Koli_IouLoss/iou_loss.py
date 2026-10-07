import tf


def iou_loss(y_true_cls,y_true_reg,y_pred_reg,coef_row,coef_col,coef_height,coef_width,img_height,img_width):

    y_true_reg[:,:,:,0]=y_true_reg[:,:,:,0]*coef_row
    y_true_reg[:,:,:,1]=y_true_reg[:,:,:,1]*coef_col
    y_true_reg[:,:,:,2]=y_true_reg[:,:,:,2]*coef_height
    y_true_reg[:,:,:,3]=y_true_reg[:,:,:,3]*coef_width

    y_pred_reg[:, :, :, 0] = y_pred_reg[:, :, :, 0] * coef_row
    y_pred_reg[:, :, :, 1] = y_pred_reg[:, :, :, 1] * coef_col
    y_pred_reg[:, :, :, 2] = y_pred_reg[:, :, :, 2] * coef_height
    y_pred_reg[:, :, :, 3] = y_pred_reg[:, :, :, 3] * coef_width


    ones_indices = tf.where(tf.reduce_any(tf.equal(y_true_cls[:,:, :, :2], 1), axis=-1))
    selected_values_reg_true = tf.gather_nd(y_true_reg, ones_indices)
    selected_values_reg_pred = tf.gather_nd(y_pred_reg, ones_indices)


    stride = tf.constant(8, dtype=tf.int64)
    half_stride = tf.constant(4, dtype=tf.int64)

    # Multiply all elements by the variable and add the constant
    center_rows = ones_indices[:,1] * stride + half_stride
    center_cols=ones_indices[:,2] * stride + half_stride

    true_w_r = selected_values_reg_true[:, 3]
    true_h_r = selected_values_reg_true[:, 2]
    true_h = tf.multiply(true_h_r, img_height)
    true_w = tf.multiply(true_w_r, img_width)

    true_y0=tf.cast(center_rows-(true_h/2), dtype=tf.int64)
    true_y=tf.add(true_y0,selected_values_reg_true[:,0])
    true_x0=tf.cast(center_cols-(true_w/2), dtype=tf.int64)
    true_x=tf.add(true_x0,selected_values_reg_true[:,1])


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

    pred_y0 = tf.cast(center_rows - (pred_h / 2), dtype=tf.int64)
    pred_y=tf.add(pred_y0,selected_values_reg_pred[:,0])
    pred_x0 = tf.cast(center_cols - (pred_w / 2),dtype=tf.int64)
    pred_x = tf.add(pred_x0, selected_values_reg_pred[:, 1])

    # max_rows_pred = tf.cast(tf.add(min_rows_pred, selected_values_reg_pred[:, 2]),dtype=tf.int64)
    # max_cols_pred = tf.cast(tf.add(min_cols_pred, selected_values_reg_pred[:, 3]),dtype=tf.int64)

    # Calculate intersection area
    inter_w = tf.math.minimum(true_x + true_w, pred_x + pred_w) - tf.math.maximum(true_x, pred_x)
    inter_h = tf.math.minimum(true_y + true_h, pred_y + pred_h) - tf.math.maximum(true_y, pred_y)
    inter_area = tf.math.maximum(0.0, inter_w) * tf.math.maximum(0.0, inter_h)

    # Calculate union area
    true_area = true_w * true_h
    pred_area = pred_w * pred_h
    union_area = true_area + pred_area - inter_area

    # Calculate GIOU
    giou = inter_area / union_area
    box_area = true_w * true_h + pred_w * pred_h

    # Calculate C (smoothing term)
    c = 1.0 - giou + (box_area - union_area) / box_area

    # Calculate GIOU loss
    giou_loss = 1.0 - giou + c



    # Calculate bounding box area
    # box_area = true_w * true_h + pred_w * pred_h

    # Calculate C (smoothing term)
    # c = 1.0 - giou

    # Calculate GIOU loss
    # giou_loss = 1.0 - giou + c

    # Calculate mean loss over all bounding boxes
    giou_loss = tf.reduce_mean(giou_loss)

    return giou_loss
