import numpy as np
from collections import Counter
from dataclasses import dataclass


@dataclass
class BBox:
    """A simple data class to hold bounding box coordinates and category."""
    min_y: int
    min_x: int
    max_y: int
    max_x: int
    category: int


def iou(boxA, boxB):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.

    Args:
        boxA (list or tuple): The first box [min_y, min_x, max_y, max_x].
        boxB (list or tuple): The second box [min_y, min_x, max_y, max_x].

    Returns:
        float: The IoU value.
    """
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = float(boxA_area + boxB_area - inter_area)

    return inter_area / union_area


def cluster_and_average_boxes(
        predicted_boxes,
        image_shape,
        minimal_iou_thr=0.4
):
    """
    Clusters a list of bounding boxes based on IoU and averages them.

    Args:
        predicted_boxes (list[list[int]]): A list of bounding boxes.
            Each box must be in the format: [min_y, min_x, max_y, max_x, class_id].
        image_shape (tuple): The shape of the input image (height, width).
        minimal_iou_thr (float): The IoU threshold for clustering detections.

    Returns:
        list[list[int]]: A list of final averaged bounding boxes in the format
                         [min_x, min_y, max_x, max_y, class_id+1].
    """
    image_height, image_width = image_shape

    # === 1. Handle edge cases (0 or 1 box) ===
    if not predicted_boxes:
        return []

    if len(predicted_boxes) == 1:
        p = predicted_boxes[0]
        # Final format: [min_x, min_y, max_x, max_y, class+1]
        final_box = [
            max(0, p[1]),
            max(0, p[0]),
            min(p[3], image_width),
            min(p[2], image_height),
            p[4] + 1
        ]
        return [final_box]

    # === 2. Cluster the positions based on IoU ===
    clusters = []
    max_classes = []
    processed_indices = [False] * len(predicted_boxes)

    for i in range(len(predicted_boxes)):
        if processed_indices[i]:
            continue

        current_cluster = []
        pom_classes = []
        box1 = predicted_boxes[i]
        processed_indices[i] = True
        current_cluster.append(BBox(*box1))
        pom_classes.append(box1[4])

        for j in range(i + 1, len(predicted_boxes)):
            if processed_indices[j]:
                continue

            box2 = predicted_boxes[j]
            if iou(box1, box2) > minimal_iou_thr:
                processed_indices[j] = True
                current_cluster.append(BBox(*box2))
                pom_classes.append(box2[4])

        if current_cluster:
            clusters.append(current_cluster)
            max_class = Counter(pom_classes).most_common(1)[0][0]
            max_classes.append(max_class)

    # === 3. Average boxes within each cluster ===
    final_coordinates = []
    for i, cluster in enumerate(clusters):
        max_class = max_classes[i]

        avg_min_y, avg_min_x, avg_max_y, avg_max_x = 0.0, 0.0, 0.0, 0.0
        count = 0

        for bbox in cluster:
            if bbox.category == max_class:
                avg_min_y += bbox.min_y
                avg_min_x += bbox.min_x
                avg_max_y += bbox.max_y
                avg_max_x += bbox.max_x
                count += 1

        if count > 0:
            avg_min_y /= count
            avg_min_x /= count
            avg_max_y /= count
            avg_max_x /= count

            avg_coord = [
                max(0, int(avg_min_x)),
                max(0, int(avg_min_y)),
                min(image_width, int(avg_max_x)),
                min(image_height, int(avg_max_y)),
                max_class + 1
            ]
            final_coordinates.append(avg_coord)

    return final_coordinates

