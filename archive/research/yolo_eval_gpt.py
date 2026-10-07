import os
import cv2
import numpy as np
from tensorflow.python.ops.metrics_impl import false_positives


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
# Paths
yolo_folder = r"D:\Monika\yolov5\runs\detect\exp11\exp11_renamed\reformatted_labels"
annot_root_path = r"D:\Monika\VideosTest\All_Cameras_annnotated\Annotations\Renamed"
image_folder = r"D:\Monika\VideosTest\All_Cameras_annnotated\Images\Renamed"
output_folder = "path_to_output_images"
os.makedirs(output_folder, exist_ok=True)

# Iterate through ground truth files
results = []
for ground_truth_folder in os.listdir(annot_root_path):
    for gt_file in os.listdir(os.path.join(annot_root_path,ground_truth_folder)):
        if gt_file.endswith(".txt"):
            image_name = gt_file.replace(".txt", ".bmp")
            image_path = os.path.join(image_folder,ground_truth_folder, image_name)
            image=cv2.imread(image_path)
            output_path = os.path.join(output_folder, image_name)

            # Load ground truths
            gt_path = os.path.join(annot_root_path,ground_truth_folder, gt_file)
            ground_truths = np.loadtxt(gt_path, delimiter=',').reshape(-1, 5) if os.path.exists(gt_path) else []

            # Load YOLO detections
            yolo_path = os.path.join(yolo_folder, gt_file[:-4]+'_'+str(ground_truth_folder)+'_reg.txt')
            detections = np.loadtxt(yolo_path, delimiter=',').reshape(-1, 5) if os.path.exists(yolo_path) else []

            # Calculate metrics

            # Reformat ground truths
            if ground_truths.size > 0:
                ground_truths = reformat_ground_truths(ground_truths)
                # print(ground_truths)
            tp, fp, fn,errors= calculate_metrics(detections, ground_truths,image)
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

# Print per-file results
for result in results:
    print(f"File: {result[0]} | TP: {result[1]} | FP: {result[2]} | FN: {result[3]}")

# Print final summary
print("\nFinal Metrics Summary:")
print(f"Total TP: {total_tp}")
print(f"Total FP: {total_fp}")
print(f"Total FN: {total_fn}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")
