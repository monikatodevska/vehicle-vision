"""
load images and ground truth annotations
parse ground truth annotations
"""

# python modules
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# custom modules


def load_video(src_path, task, dst_path=None, name_prefix=None):
    """
    load video and iterate through frames
    requirements: video codecs (ffmpeg)
    :param src_path: global video path including file name [string]
    :param task: determines action on the extracted video frames (0 - view, 1 - save) [int]
    :param dst_path: destination folder to save extracted frames [string]
    :param name_prefix: name prefix for saved frames [sting]
                        name format: prefix_000001.bmp
    :return: None
    """

    # load video
    vid_cap = cv2.VideoCapture(src_path)  # create video capture object
    success, frame = vid_cap.read()  # load first frame, populate success variable

    # extract frames
    frame_cnt = 0  # frame counter

    while success:

        if task == 0:
            # display frame
            cv2.imshow('frame', frame)
            cv2.waitKey(0)

        if task == 1:
            # save frame
            frame_name = name_prefix + '_' + str(frame_cnt).zfill(6) + '.bmp'  # create file name
            cv2.imwrite(os.path.join(dst_path, frame_name), frame)  # save frame as PNG file

        # read new frame
        success, frame = vid_cap.read()  # load next frame
        print(f'Frame number {frame_cnt}. Success: ', success)
        frame_cnt += 1


def load_images_single_camera(src_path, crop_coords, output_dims, output_depth):
    """
    load images of one view of one scene
    :param src_path: parent directory for one view and one camera [string]
    :param crop_coords: coordinates to crop image ((min_row, min_col + 1), (max_row, max_col + 1)) [tuple]
    :param output_dims: size of output images (height, width) [tuple]
    :param output_depth: number of image channels (1 - grayscale, 3 - BGR) [int]
    :return: array of loaded images [ndarray]
             dimensions: num_images, height, width, depth
    """

    images = []

    img_names = [x for x in os.listdir(src_path) if os.path.splitext(x)[1] == '.bmp']

    for img_name in tqdm(img_names):

        if output_depth == 1:
            img = cv2.imread(os.path.join(src_path, img_name), 0)
        else:
            img = cv2.imread(os.path.join(src_path, img_name))

        if img is None:
            print(f'Image name {os.path.join(src_path, img_name)} could not be read.')
            continue

        img = img[crop_coords[0][0]:crop_coords[0][1], crop_coords[1][0]:crop_coords[1][1]]
        img = cv2.resize(img, (output_dims[1], output_dims[0]), interpolation=cv2.INTER_CUBIC)

        images.append(img)

    images = np.array(images)

    return images


def load_images_multi_camera(src_path, crop_coords, output_dims, output_depth):
    """
    load images of all views of one scene
    :param src_path: src_path: parent directory containing all views of one scene [string]
    :param crop_coords: coordinates to crop image ((min_row, min_col + 1), (max_row, max_col + 1)) [tuple]
    :param output_dims: size of output images (height, width) [tuple]
    :param output_depth: number of image channels (1 - grayscale, 3 - BGR) [int]
    :return: array of loaded images grouped by view [ndarray]
             dimensions: num_views, num_images, height, width, depth
    """

    images_scene = []

    cam_names = [x for x in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, x))]

    for cam_name in cam_names:
        src_path_cam = os.path.join(src_path, cam_name)
        images = load_images_single_camera(src_path_cam, crop_coords, output_dims, output_depth)
        images_scene.append(images)

    images_scene = np.array(images_scene)

    return images_scene


def load_parse_labels_single_camera(src_path):
    """
    loads and names label data to pandas dataframe
    NOTE: columns X1 to X4 contain no meaningful data, placeholder for real world positions
    :param src_path: global path including file name to file containing labels for one camera of one view [string]
    :return: pandas dataframe containing named label data [dataframe]
    """

    label_data = np.loadtxt(src_path, delimiter=',').astype(np.int)

    labels = pd.DataFrame(label_data, columns=['FrameNum', 'ObjID', 'MinC', 'MinR', 'Width', 'Height',
                                               'X1', 'X2', 'X3', 'X4'])

    return labels


def load_parse_labels_multi_camera(src_path):
    """

    :param src_path:
    :return:
    """
    pass


# function tests
if __name__ == '__main__':

    # load video

    src_path = r'\\192.168.1.148\Public\Datasets\AICityChallenge2023\AI_track1\train\S002\c008\video.mp4'
    task = 1
    dst_path = r'\\192.168.1.148\Public\Datasets\AICityChallenge2023\AI_track1\train\S002\c008\frames'
    name_prefix = 'frame'

    load_video(src_path, task, dst_path, name_prefix)

    # load frames
    src_path = r'\\192.168.1.148\Public\Datasets\AICityChallenge2023\AI_track1\train\S002\c008\frames'
    crop_coords = ((0, 1080 + 1), (0, 1920 + 1))
    output_dims = (1080, 1920)
    output_depth = 3
    images = load_images_single_camera(src_path, crop_coords, output_dims, output_depth)
    print(images.shape)
    cv2.imshow('img', images[0])
    cv2.waitKey(0)

    # load and parse input data
    src_path = r'\\192.168.1.148\Public\Datasets\AICityChallenge2023\AI_track1\train\S002\c008\label.txt'
    labels = load_parse_labels_single_camera(src_path)
    print(labels.head())
