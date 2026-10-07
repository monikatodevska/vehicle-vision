import os
import cv2
import numpy as np
# from tensorflow.python.ops.metrics_impl import false_positives
import helper_model,helper_postprocessing
debug=True
def compute_iou(box1, box2):
    """
    Computes the Intersection over Union (IoU) of two bounding boxes.
    box format: [ymin, xmin, ymax, xmax]
    """
    ymin1, xmin1, ymax1, xmax1 = box1
    ymin2, xmin2, ymax2, xmax2 = box2

    inter_ymin = max(ymin1, ymin2)
    inter_xmin = max(xmin1, xmin2)
    inter_ymax = min(ymax1, ymax2)
    inter_xmax = min(xmax1, xmax2)

    inter_area = max(0, inter_ymax - inter_ymin) * max(0, inter_xmax - inter_xmin)
    box1_area = (ymax1 - ymin1) * (xmax1 - xmin1)
    box2_area = (ymax2 - ymin2) * (xmax2 - xmin2)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def calc_iou(box1, box2):
    """

    :param box1: list of coordinates: row1, col1, row2, col2 [list]
    :param box2: list of coordinates: row1, col1, row2, col2 [list]
    :return: iou value
    """

    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # respective area of the two boxes
    boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # overlap area
    interArea = max(xB - xA, 0) * max(yB - yA, 0)

    # IOU

    if (boxAArea+boxBArea-interArea) > 0:
        # print(box1, box2)
        iou = interArea / (boxAArea + boxBArea - interArea)
        return iou

    print("union < 0")
    # return
    exit(1)


def adjust_coordinates(detections, original_size, new_size, crop_left):
    original_width, original_height = original_size
    new_width, new_height = new_size

    # Scaling factors
    width_scale = 1620 / 512
    height_scale = 1080 / 341

    adjusted_detections = []

    for bbox in detections:
        # Apply left crop shift
        xmin=bbox[1]
        ymin=bbox[0]
        xmax=bbox[3]
        ymax=bbox[2]

        # Rescale coordinates to the new image size
        new_xmin = int(xmin * width_scale)
        new_xmax = int(xmax * width_scale)
        new_ymin = int(ymin * height_scale)
        new_ymax = int(ymax * height_scale)

        new_xmin += crop_left
        new_xmax += crop_left
        adjusted_detections.append([new_ymin, new_xmin, new_ymax, new_xmax,bbox[4]])

    return adjusted_detections



def getclusterforanchor(anchor, lista, thr):
    cluster = []
    for anch in lista:
        # print(anch)
        iou = calc_iou(anchor, anch)
        # print(round(iou,1))
        if iou > thr:
            cluster.append(anch)
    # print("length clys", len(cluster))
    return cluster


def findelement(maxnum, cluster):
    output = []
    for item in cluster:
        if item[0] == maxnum:
            output.append(item)

    if len(output) == 0:
        print("Imas bug vo baranjeto maximum")
        exit(1)
    return output


def average_window(maxwindows):
    # print(maxwindows)
    maxcoord = maxwindows[0][0]
    for window in maxwindows:
        for x in window:
            if maxcoord>x:
                maxcoord=x
    # print(maxcoord)
    suma = [sum(x) for x in zip(*maxwindows)]
    avg = [x / len(maxwindows) for x in suma]

    # print(suma,avg)
    return avg



def getclustersforanchors(lista, thr):
    clusters = []
    res = []
    finalclusters = []
    countconnections = []
    # print(lista)

    for anchor in lista:


        # print(anchor)
        cluster = getclusterforanchor(anchor, lista, thr)
        # print(cluster)
        countconnections.append(len(cluster))
        clusters.append(cluster)
    maxnum = countconnections[0]
    cnt = -1
    indexestoremove = []
    # print("countconnections")
    # print(len(countconnections))
    # print(countconnections)
    if (len(clusters) == 1):
        return clusters
    for anchor in lista:
        clusters_idx = []
        # connection_sublist = []
        # print("anchor", anchor)
        # print("clusters", clusters)
        for cluster in clusters:
            # print(cluster)
            if anchor in cluster:
                clusters_idx.append(1)
                # connection_sublist.append(countconnections[cnt])
            else:
                clusters_idx.append(0)

        indices = [i for i, x in enumerate(clusters_idx) if x == 1]
        # print(len(clusters_idx))
        # print(len(clusters))
        # print(indices)
        maxcluster = clusters[indices[0]]
        # print(maxcluster)
        for indx in indices:

            # print(indx,clusters[indx])
            # print(maxcluster,len(maxcluster))
            if (len(clusters[indx]) > len(maxcluster)):
                maxcluster = clusters[indx]
        res.append(maxcluster)

    # print("res", res)
    finalclusters = []
    for i in res:
        if i not in finalclusters:
            finalclusters.append(i)

    # print("finalclusters", finalclusters)

    return finalclusters

def nms_new(lista, thr, num_classes):
    finalwindows = []

    if len(lista) == 0:
        return []

    all_clusters = getclustersforanchors(lista, thr)

    for cluster in all_clusters:

        maxwindows = []
        maxwindows_new = []

        if len(cluster) == 0:
            continue

        elements = [item[2] - item[0] for item in cluster]

        if len(elements) <= 2:
            first2max = sorted(elements)
        else:

            first2max = sorted(elements)[:]

        first2max_noduplicates = []
        for i in first2max:
            if i not in first2max_noduplicates:
                first2max_noduplicates.append(i)

        for maxnum in first2max_noduplicates:
            indices = [i for i, x in enumerate(elements) if x == maxnum]

            for idx in indices:
                maxwindows.append(cluster[idx])

        bboxes_per_class = [[] for i in range(num_classes)]  # bounding boxes grouped by class

        for maxwindow in maxwindows:
            bboxes_per_class[maxwindow[-1]].append(maxwindow)

        max_class = max((len(l), i) for i, l in enumerate(bboxes_per_class))[1]

        for maxwindow in maxwindows:
            if maxwindow[-1] == max_class:
                maxwindows_new.append(maxwindow)

        finalwindow = average_window(maxwindows_new)
        finalwindows.append(finalwindow)

    return finalwindows

def preprocess_data(image,dir_name):

    rows, cols = image.shape[:2]

    if 'videoframes' in dir_name:
        cropped_image = image[0:rows, 0:1141]
        left_crop = 0

    elif 'miladinovci' in dir_name:
        cropped_image = image[0:rows, 0:1621]
        left_crop = 0

    elif 'kamera2' in dir_name or 'kam23' in dir_name:
        cropped_image = image[0:rows, 160:1782]
        left_crop = 160


    elif ('M-30' in dir_name) or ('drugo' in dir_name):
        # break
        # print('m30')
        pass

    elif 'mil_res' in dir_name:
        pass

    elif 'kam25' in dir_name:
        cropped_image = image[0:rows, 0:1621]
        left_crop=0

    elif 'kam28' in dir_name:
        cropped_image = image[0:rows, 0:1621]
        left_crop = 0
    elif 'kam30' in dir_name:
        cropped_image = image[0:rows, 0:1621]
        left_crop = 0
    elif 'kam32' in dir_name:
        cropped_image = image[0:rows, 299:cols]
        left_crop=299
    elif 'kam33' in dir_name:
        cropped_image = image[0:rows, 99:1720]
        left_crop = 99
    elif 'kam36' in dir_name:
        cropped_image = image[0:rows, 189:1810]
        left_crop = 189
    elif 'kam40' in dir_name:
        cropped_image = image[0:rows, 255:255 + 1620]
        left_crop = 255
    elif 'kam42' in dir_name:
        cropped_image = image[0:rows, 299:cols]
        left_crop = 299
    elif 'kam44' in dir_name:
        cropped_image = image[0:rows, 299:cols]
        left_crop =299
    elif 'kam46' in dir_name:
        cropped_image = image[0:rows, 100:1721]
        left_crop = 100
    elif 'kam48' in dir_name:
        cropped_image = image[0:rows, 299:cols]
        left_crop = 299


    resized = cv2.resize(cropped_image, (512, 341), interpolation=cv2.INTER_AREA)
    return resized,left_crop
def save_results_anchorless_limits(number_of_images, output_cls, output_reg, anchor_stride, prob_thr,
                                   thr_clustering):
    """
    plot bounding boxes of detected objects onto test images and save as images
    :param results_path: path of destination folder for all CNN detections [string]
    :param results_path_nms: path of destination folder of prostprocessed data [string]
    :param images: test images [ndarray]
    :param output_cls: output of the classifier [ndarray]
    :param output_reg: output of the regressor [ndarray]
    :param anchor_stride: stride along rows and columns [int]
    :param prob_thr: probability threshold for object classification (range: 0 to 1) [float]
    :param norm_coef_position: normalization coefficient for position regression data [float]
    :param norm_coef_size: normalization coefficient for image dimensions regression data [float]
    :param thr_clustering: iou threshold for combining detections into one cluster [float]
    :param colors_list: list of colors to plot each class [tuple]
    :return: None
    """
    img_dims = (341, 512)  # height, width
    # print(images_s[0].shape[0])
    num_classes = output_cls.shape[3] - 1

    # binarize classifier output probabilities
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = np.int(np.round(anchor_stride / 2))
    start_c = np.int(np.round(anchor_stride / 2))
    finalwindows_batch = []

    for im_ind in range(number_of_images):

        # image_za_nms = deepcopy(image)
        valid_bboxes = []  # list of bounding boxes with positive height and width

        res = output_cls[im_ind, :, :, :-1]  # classifier output, probability maps for positive objects only

        [r, c, d] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)

        for pred_ind in range(len(r)):  # iterate over positive predictions
            # ako e klasa nevozilo ne crtaj i ne zacucuvaj koordinati
            if d[pred_ind] == 2:
                continue
            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # 4 - 4 dimensions are fine-tuned: r, c (top left corner), h, w
            delta_r = output_reg[im_ind, r[pred_ind], c[pred_ind], 0]
            delta_c = output_reg[im_ind, r[pred_ind], c[pred_ind], 1]
            h_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 2]
            w_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 3]

            h = h_percent * img_dims[0]
            w = w_percent * img_dims[1]
            # print(h)
            # print(w)
            # bbox top left point
            min_row = np.int(center_row - np.round(h / 2))
            min_col = np.int(center_col - np.round(w / 2))

            # adjust position and size with regressor predictions
            min_row_adj = np.int(min_row + delta_r)
            min_col_adj = np.int(min_col + delta_c)

            max_row_adj = min_row_adj + h
            max_col_adj = min_col_adj + w

            # plot bounding box onto image
            # clip bounding boxes falling out of the image borders
            min_col_adj = max(0, np.int(min_col_adj))
            min_row_adj = max(0, np.int(min_row_adj))
            max_col_adj = min(np.int(max_col_adj), img_dims[1])
            max_row_adj = min(np.int(max_row_adj), img_dims[0])

            if (max_row_adj - min_row_adj) > 0 and (max_col_adj - min_col_adj) > 0:
                valid_bboxes.append([min_row_adj, min_col_adj, max_row_adj, max_col_adj, d[pred_ind]])

        if len(r) != 0:

            finalwindows = nms_new(valid_bboxes, thr_clustering, num_classes)

            # finalwindows_batch.append(finalwindows)

        else:

            finalwindows=[]

    return finalwindows
def reformat_ground_truths(ground_truths):
    """
    Reformat ground truths from (ymin, xmin, h, w, class) to (ymin, xmin, ymax, xmax, class).

    Args:
        ground_truths (np.ndarray): Array of ground truths in the format (ymin, xmin, h, w, class).

    Returns:
        np.ndarray: Reformatted ground truths in the format (ymin, xmin, ymax, xmax, class).
    """
    reformatted = ground_truths.copy()
    reformatted[:, 2] = reformatted[:, 0] + reformatted[:, 2]  # ymax = ymin + h
    reformatted[:, 3] = reformatted[:, 1] + reformatted[:, 3]  # xmax = xmin + w
    return reformatted
def calculate_metrics_best_iou(ground_truths,detections,im_orig,im_path_dest,iou_threshold=0.3):
    matched_detections = set()
    matched_ground_truths = set()
    errors = []

    # Pair detections with ground truths based on the best IoU
    iou_matrix = np.zeros((len(ground_truths), len(detections)))

    # Compute IoU for all pairs
    for i, gt in enumerate(ground_truths):
        for j, det in enumerate(detections):
            iou_matrix[i, j] = compute_iou(gt[:4], det[:4])

    # Process matches
    while True:
        # Find the maximum IoU in the matrix
        if iou_matrix.size > 0:
            max_iou = np.max(iou_matrix)
        else:
            print('losa matrica')
            max_iou=0

        # Break if the maximum IoU is below the threshold
        if max_iou < iou_threshold:
            break

        # Get the indices of the maximum IoU
        gt_idx, det_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)

        # Match this detection and ground truth
        matched_detections.add(det_idx)
        matched_ground_truths.add(gt_idx)

        # Calculate center point error
        gt_center = [(ground_truths[gt_idx][1] + ground_truths[gt_idx][3]) / 2, ground_truths[gt_idx][2]]
        det_center = [(detections[det_idx][1] + detections[det_idx][3]) / 2, detections[det_idx][2]]
        error = np.sqrt((gt_center[0] - det_center[0]) ** 2 + (gt_center[1] - det_center[1]) ** 2)
        errors.append(error)

        # Invalidate the matched rows and columns
        iou_matrix[gt_idx, :] = -1  # Invalidate this ground truth (row)
        iou_matrix[:, det_idx] = -1  # Invalidate this detection (column)

    tp = len(matched_detections)
    fp = len(detections) - tp
    fn = len(ground_truths) - len(matched_ground_truths)

    false_positives = [detections[j] for j in range(len(detections)) if j not in matched_detections]
    false_negatives = [ground_truths[i] for i in range(len(ground_truths)) if i not in matched_ground_truths]

    # Visualize false positives and false negatives on the image
    if debug:
        for gt in ground_truths:
            ymin, xmin, ymax, xmax, _ = gt
            cv2.rectangle(im_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)  # Red for FP

        for det in false_positives:
            ymin, xmin, ymax, xmax, _ = det
            cv2.rectangle(im_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)  # Red for FP
            cv2.putText(im_orig, "FP", (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


        for gt in false_negatives:
            ymin, xmin, ymax, xmax, _ = gt
            cv2.rectangle(im_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)  # Blue for FN
            cv2.putText(im_orig, "FN", (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        if len(false_positives) > 0 or len(false_negatives) > 0:
            cv2.imwrite(im_path_dest,im_orig)

    return tp, fp, fn, errors
def calculate_metrics(detections, ground_truths, image,iou_threshold=0.3):
    """
    Pairs detections with ground truths and calculates FP, FN, precision, recall, F1-score, and mean error.
    """
    matched_detections = set()
    matched_ground_truths = set()
    errors = []

    # Pair detections with ground truths based on IoU
    for i, gt in enumerate(ground_truths):
        for j, det in enumerate(detections):
            if j not in matched_detections and i not in matched_ground_truths:
                iou = compute_iou(gt[:4], det[:4])
                if iou >= iou_threshold:
                    matched_detections.add(j)
                    matched_ground_truths.add(i)
                    # Calculate center point error
                    gt_center = [(gt[1] + gt[3]) / 2, gt[2]]  # Center of the down line
                    det_center = [(det[1] + det[3]) / 2, det[2]]
                    error = np.sqrt((gt_center[0] - det_center[0]) ** 2 + (gt_center[1] - det_center[1]) ** 2)
                    errors.append(error)

    # Calculate metrics
    fp = len(detections) - len(matched_detections) #kolku od vkupniot broj na detekcii bile spareni, ostanatite se fp
    fn = len(ground_truths) - len(matched_ground_truths)  #kolku od vkupniot broj na ground truths bile spareni,ostananatite se fn
    tp = len(matched_detections)
    false_positives=[]
    false_negatives=[]
    for j, det in enumerate(detections):
        if j not in matched_detections:  # If the detection was not matched with any ground truth
            false_positives.append(det)

    for i, gt in enumerate(ground_truths):
        if i not in matched_ground_truths:  # If the ground truth was not matched with any detection
            false_negatives.append(gt)
        # Visualize the False Positives on the image
    # image = cv2.imread(image)  # Read the image

    # Draw false positives in red on the image
    if debug:
        for det in false_positives:
            ymin, xmin, ymax, xmax, _ = det
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)  # Red color for FP
            cv2.putText(image, "FP", (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        for gt in false_negatives:
            ymin, xmin, ymax, xmax, _ = gt
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)  # Blue color for FN
            cv2.putText(image, "FN", (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        if len(false_positives)>0 or len(false_negatives)>0:
            cv2.imshow('s',image)
            cv2.waitKey(0)


    # precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    # recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    # f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # mean_error = np.mean(errors) if errors else None

    return tp, fp, fn,errors


def visualize_results(image_path, detections, ground_truths, output_path):
    """
    Visualizes paired detections and ground truths on the image.
    """
    image = cv2.imread(image_path)

    for det in detections:
        ymin, xmin, ymax, xmax, cls = det
        color = (0, 255, 0)  # Green for detections
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        cv2.putText(image, f"Det:{int(cls)}", (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for gt in ground_truths:
        ymin, xmin, ymax, xmax, cls = gt
        color = (255, 0, 0)  # Blue for ground truth
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        cv2.putText(image, f"GT:{int(cls)}", (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imshow('sl', image)



# Paths
yolo_folder = r"D:\Monika\yolov5\runs\detect\exp11\exp11_renamed\reformatted_labels"
annot_root_path = r"D:\Monika\VideosTest\All_Cameras_annnotated\Annotations\Renamed"
image_folder = r"D:\Monika\VideosTest\All_Cameras_annnotated\Images\Renamed"
output_folder = "M:\Monika\magisterska\Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor1_All_Negatives_Renamed_diou_focal_notnormreg_anotiraniguzva"
os.makedirs(output_folder, exist_ok=True)
srcModelPath=r'D:\Monika\Models\Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor1_All_Negatives_Renamed_diou_focal_notnormreg_anotiraniguzva'
max_prob_thr=1
# Example usage
original_size = (512, 341)  # original image size (after crop and before resize)
new_size = (1920, 1080)  # target size





#Load model
model = helper_model.load_model(model_path=os.path.join(srcModelPath,'model.json'),
                                    weights_path=os.path.join(srcModelPath, 'model.h5'))  # build model architecture

# Iterate through ground truth files
frame_batch=1
anchor_stride=8
thr_clustering=0.3
best_f1 = 0
best_thr=0
for prob_thr in np.arange(0.7, 0.75, 0.05):
    results = []

    print(prob_thr)
    num_vehicles=0
    for ground_truth_folder in os.listdir(annot_root_path):
        # if not 'kam33_1' in ground_truth_folder:
        #     continue
        for gt_file in os.listdir(os.path.join(annot_root_path,ground_truth_folder)):
            if gt_file.endswith(".txt"):
                image_name = gt_file.replace(".txt", ".bmp")
                image_path = os.path.join(image_folder,ground_truth_folder, image_name)
                im_orig=cv2.imread(image_path,1)
                image=cv2.imread(image_path,0)
                image_res,left_crop=preprocess_data(image,ground_truth_folder)
                im_path_dest = os.path.join(output_folder, image_name)

                # Load ground truths
                gt_path = os.path.join(annot_root_path,ground_truth_folder, gt_file)
                ground_truths = np.loadtxt(gt_path, delimiter=',').reshape(-1, 5) if os.path.exists(gt_path) else []
                num_vehicles+=len(ground_truths)
                image_reshaped=np.reshape(image_res,image_res.shape+(1,))
                image_reshaped = np.expand_dims(image_reshaped, axis=0)
                output_combined = model.predict(image_reshaped, verbose=1)
                output_cls=output_combined[:,:,:,:4]
                output_reg=output_combined[:,:,:,4:]
                detections_model = save_results_anchorless_limits(frame_batch, output_cls,
                                                                                       output_reg,
                                                                                       anchor_stride, prob_thr,
                                                                                       thr_clustering
                                                                                       )

                detections=adjust_coordinates(detections_model,original_size,new_size,left_crop)
                # Load YOLO detections
                # for bbox in detections:
                #     cv2.rectangle(image,(bbox[1],bbox[0]),(bbox[3],bbox[2]),color=(255,0,0),thickness=1)
                # cv2.imshow('s',image)
                # cv2.waitKey(0)
                # yolo_path = os.path.join(yolo_folder, gt_file[:-4]+'_'+str(ground_truth_folder)+'_reg.txt')
                # detections = np.loadtxt(yolo_path, delimiter=',').reshape(-1, 5) if os.path.exists(yolo_path) else []

                # Calculate metrics

                # Reformat ground truths
                if ground_truths.size > 0:
                    ground_truths_ref = reformat_ground_truths(ground_truths)
                    # num_vehicles+=1
                    # print(ground_truths)
                tp, fp, fn,errors= calculate_metrics_best_iou(ground_truths_ref,detections,im_orig,im_path_dest,iou_threshold=0.3)
                results.append((gt_file, tp, fp, fn,errors))

        # Visualize results
        # visualize_results(image_path, detections, ground_truths, output_path)
# Print summary
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_error = 0
    total_error_count = 0
    # Iterate through results to compute totals
    for result in results:
        total_tp += result[1]
        total_fp += result[2]
        total_fn += result[3]
        total_error += sum(result[4])  # Sum of errors for the file
        total_error_count += len(result[4])

    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    if f1_score >= best_f1:
        best_f1 = f1_score
        best_thr = prob_thr
        best_precision=precision
        best_recall=recall
        fps=total_fp
        fns=total_fn
        tps=total_tp

    # Print per-file results
    # for result in results:
    #     print(f"File: {result[0]} | TP: {result[1]} | FP: {result[2]} | FN: {result[3]}")

    # Print final summary
    print("\nFinal Metrics Summary:")
    print(f"Total TP: {total_tp}")
    print(f"Total FP: {total_fp}")
    print(f"Total FN: {total_fn}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")

print("BEST")
# print(best_thr)
print("\nFinal Metrics Summary:")
print(f"Total TP: {tps}")
print(f"Total FP: {fps}")
print(f"Total FN: {fns}")
print(f"Precision: {best_precision:.2f}")
print(f"Recall: {best_recall:.2f}")
print(f"F1 Score: {best_f1:.2f}")
print(f"Best thr: {best_thr:.2f}")
print(f"Num_vehicles: {num_vehicles:.2f}")