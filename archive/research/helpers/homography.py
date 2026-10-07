import cv2
import numpy as np
import json
import os


def load_config(config_directory, camera_name):
    config_path = os.path.join(config_directory, camera_name, "config_camera.json")
    with open(config_path, 'r') as file:
        return json.load(file)


def homography_matrix_create(config_directory, camera_name, orig_img_height, orig_img_width, img_h_vehicle_tracking,
                             img_w_vehicle_tracking):
    # Load config from the appropriate directory
    config = load_config(config_directory, camera_name)

    # Extracting relevant parameters
    real_time_pixels = config["MarkersInImage"]
    projection_dims = config["MarkersInWorld"]
    crop_rect = list(map(int, config["CropRect"].split(",")))

    br_tocki = len(real_time_pixels) // 2

    crop_top = crop_rect[1]
    crop_left = crop_rect[0]
    crop_bottom = orig_img_height - crop_rect[3] - crop_rect[1]
    crop_right = orig_img_width - crop_rect[2] - crop_rect[0]

    # Convert real-time pixels into a list of (x, y) tuples
    pts_src = np.array([(real_time_pixels[i], real_time_pixels[i + 1]) for i in range(0, len(real_time_pixels), 2)],
                       dtype=np.float32)

    # Convert real-world points to meters
    pts_dst = np.array(
        [(projection_dims[i] / 1000.0, projection_dims[i + 1] / 1000.0) for i in range(5, len(projection_dims), 2)],
        dtype=np.float32)

    # Compute resize coefficients
    resize_coef_w = (orig_img_width - crop_left - crop_right) / img_w_vehicle_tracking
    resize_coef_h = (orig_img_height - crop_top - crop_bottom) / img_h_vehicle_tracking

    # Adjust source points according to cropping and resizing
    pts_src[:, 0] = (pts_src[:, 0] - crop_left) / resize_coef_w
    pts_src[:, 1] = (pts_src[:, 1] - crop_top) / resize_coef_h

    # Compute homography matrix
    h, _ = cv2.findHomography(pts_src, pts_dst)

    return h.astype(np.float32)

def image2ProjPoint(img_point,h):
    point = np.array([img_point[0], img_point[1], 1], dtype=np.float32).reshape(3, 1)
    transformed_point=np.dot(h,point)
    transformed_point=transformed_point/transformed_point[2,0]

    return transformed_point[0,0], transformed_point[1, 0]