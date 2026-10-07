import cv2
import numpy as np
import os

src_path=r''
def read_bboxes(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    bboxes = []
    for line in lines:
        y1, x1, y2, x2 = map(int, line.strip().split())
        bboxes.append((y1, x1, y2, x2))
    return bboxes

def crop_vehicles(image, bboxes, size=(48, 48)):
    patches = []
    h_img, w_img = image.shape[:2]

    for y1, x1, y2, x2 in bboxes:
        h = y2 - y1
        w = x2 - x1
        center_y = (y1 + y2) // 2
        center_x = (x1 + x2) // 2
        side = max(h, w)

        # Compute new square box
        new_y1 = max(0, center_y - side // 2)
        new_y2 = min(h_img, center_y + side // 2)
        new_x1 = max(0, center_x - side // 2)
        new_x2 = min(w_img, center_x + side // 2)

        # Ensure final square patch is square
        patch = image[new_y1:new_y2, new_x1:new_x2]
        patch = cv2.resize(patch, size)
        patches.append(patch)

    return patches
def get_similarity_scores(patches1, patches2, siamese_model, threshold=0.5):
    matched_pairs = []
    for i, patch1 in enumerate(patches1):
        best_match_idx = -1
        best_score = 0
        for j, patch2 in enumerate(patches2):
            # Siamese expects a pair
            pair = [np.expand_dims(patch1, axis=0), np.expand_dims(patch2, axis=0)]
            score = siamese_model.predict(pair)[0][0]  # assuming model outputs similarity
            if score > threshold and score > best_score:
                best_score = score
                best_match_idx = j
        if best_match_idx != -1:
            matched_pairs.append((i, best_match_idx, best_score))
    return matched_pairs

def match_frames(frame1_path, frame2_path, txt1_path, txt2_path, siamese_model):
    # Load images
    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)

    # Read bboxes
    bboxes1 = read_bboxes(txt1_path)
    bboxes2 = read_bboxes(txt2_path)

    # Crop vehicle patches
    patches1 = crop_vehicles(frame1, bboxes1)
    patches2 = crop_vehicles(frame2, bboxes2)

    # Normalize patches (optional depending on your model's preprocessing)
    patches1 = np.array(patches1) / 255.0
    patches2 = np.array(patches2) / 255.0

    network = helper_model.load_branch_structure(src_branch_structure_path)
    model = helper_model.siamese_build_model(input_dims, network)
    # predictions = model.predict([inputs1_test, inputs2_test])

    # Match vehicles
    matches = get_similarity_scores(patches1, patches2, siamese_model)

    return matches





frame_paths=os.listdir(src_path)
for ind,f_path in enumerate(frame_paths):
    frame1_path=os.path.join(src_path,f_path)
    frame2_path=os.path.join(src_path,frame_paths[ind+1])
    txt_path1=os.path.join(src_path,f_path[:-4]+'.txt')
    txt_path2=os.path.join(src_path,frame_paths[ind+1]+'.txt')
    siamese_model=
    match_frames(frame1_path,frame2_path,txt_path1,txt_path2,siamese_model)

def match_frames(frame1_path, frame2_path, txt1_path, txt2_path, siamese_model):
    # Load images
    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)

    # Read bboxes
    bboxes1 = read_bboxes(txt1_path)
    bboxes2 = read_bboxes(txt2_path)

    # Crop vehicle patches
    patches1 = crop_vehicles(frame1, bboxes1)
    patches2 = crop_vehicles(frame2, bboxes2)

    # Normalize patches (optional depending on your model's preprocessing)
    patches1 = np.array(patches1) / 255.0
    patches2 = np.array(patches2) / 255.0

    # Match vehicles
    matches = get_similarity_scores(patches1, patches2, siamese_model)
    return matches
