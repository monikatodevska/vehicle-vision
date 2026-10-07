"""
Course: Mashinski vid, FEEIT, Spring 2021
Date: 19.04.2022

Description: apply a fully convolutional SSD architecture for object classification and localization to test data
Python version: 3.6
"""

# python imports
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from keras.optimizers import Adam

# custom package imports
import helper_model, helper_data, helper_losses
import anchorless_genertor_plot


# --- flags ---
flag_save_intermediate_output = False


# --- paths ---
# version = 'vinf_400_cls_reg_finetune_cel_so_nokni_samrak_kamera2_vtor_del'
version = r'vinf_900_SO_DVA_INCEPTION_SKIP_48_normalizedByRegion_v1_Morning_5fps'

# NOTE: specify destination paths
srcImagesPath = r'D:\Monika\VideosTest\Frames_morning_5fps'
# srcImagesPath = r'D:\Monika\VideosTest\vtor_del'
# srcImagesPath = r'\\HP-1060\ForSharingD\VideoFrames\Samrak'
# srcImagesPath = r'D:\Monika\VideosTest\Frames_Utrinsko'
# srcImagesPath = r'\\192..0 168.1.153\d\VideosTest\Frames_Fleki'
# srcImagesPath = r'\\HP-1060\ForSharingD\video3_frames_black'
# srcAnnotationsPathTest = r'E:\Science\Monika4\GT_test'

src_model_version = r'vinf_900_SO_DVA_INCEPTION_SKIP_48_normalizedByRegion_v1'   #NOTE: da se pishe
srcModelPath = os.path.join(r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Models',src_model_version)
# srcModelPath = os.path.join(r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Models', src_model_version, 'new')

dstResultsPath = r'D:\Monika\Results'
# gtDstPath = r'E:\Science\Monika\GT'
# srcAnnotationsPathTest = r'E:\Science\Monika\anchorless_test'
file_path_reg_coef = os.path.join(r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Models', src_model_version, 'reg_coef')

# create folders to save data from the current execution
if not os.path.exists(os.path.join(dstResultsPath, version)):
    os.mkdir(os.path.join(dstResultsPath, version))
resultsPath = os.path.join(dstResultsPath, version)

results_path_nms = os.path.join(dstResultsPath, version + '_postprocessing')
if not os.path.exists(results_path_nms):
    os.mkdir(results_path_nms)

# --- variables ---
imgDims = {'rows': 341, 'cols': 512}
num_classes = 2
img_depth = 3
img_dims = (imgDims['rows'], imgDims['cols'], img_depth)


# --- load and format data ---
# load full dataset into memory - image data and labels
x_test, images_names = helper_data.read_data_test(srcImagesPath, (imgDims['cols'], imgDims['rows']), img_depth)

print(f'Number of test samples: {x_test.shape[0]}')


# --- prepare ground truth data in required format ---

anchor_stride = 8
out_class_list=[]
# for filename in os.listdir(srcAnnotationsPathTest):
#     fid1 = open(os.path.join(srcAnnotationsPathTest, filename), 'rb')
#     out_class_dims = pickle.load(fid1)
#     out_class_back = pickle.load(fid1)
#     fid1.close()
#     out_class = np.reshape(out_class_back, (out_class_dims[0], out_class_dims[1], out_class_dims[2]))
#     out_class_list.append(out_class)
# out_class_list = np.array(out_class_list)

# construct model ---
lr = 0.0001
model = helper_model.load_model(model_path=os.path.join(srcModelPath,'model.json'),
                                weights_path=os.path.join(srcModelPath, 'model.h5'))  # build model architecture

# compile model
model.compile(loss={
                  'out_class': helper_losses.rpn_loss_cls


                   },
              optimizer=Adam(lr=lr),
              metrics=['accuracy'])

x_test_1 = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in x_test]
x_test_1 = np.array(x_test_1)
x_test_1 = x_test_1.reshape(x_test_1.shape + (1,))

regions=[]
norm_coeff=[]
f=open(os.path.join(file_path_reg_coef, 'reg_norm_coeff.txt'),'r')
lines=f.readlines()
characters_to_remove=['[', ']']
lines_stripped=[]
lines_p=[]
list_all=[]
for ind, line in enumerate(lines):
    new_list = []
    line=line.rstrip()
    line=line[1:-1]
    k=line.split(',')
    for char in k:
        # print(char)
        if (char!='[' and char!=']' and char!=','):
            if ind==5 or ind==7:
                char=float(char)
            else:
                char=int(char)
            new_list.append(char)
    # print(new_list)
    list_all.append(new_list)
print(list_all)
for k in range(0,len(list_all)-1,2):
    regions.append(list_all[k])
    norm_coeff.append(list_all[k+1])
# print(regions)
# print(norm_coeff)
dest_matrix=[0,1,2,3]
# for line in lines:
#     print(line[1:-2])
#     lines_p.append(line[1:-2])
#     # print(line)
#     # for character in characters_to_remove:
#     #     line=line.replace(character,"")
#     # line=line.rstrip()
#     # lines_stripped.append(line)
#         # print(line)
# print(lines_p)
# for x in range(0,(len(lines_p))-1,2):
#     # print(lines_stripped)
#     # print(lines_stripped[x])
#     # print(lines_stripped[x][0])
#     for i in range(0,(len(lines_p[x]))-1):
#          # regions=[[int(str(j)) for j in i.split(',')] for i in lines_p[x]]
#         print(lines_p[x][i])
#         regions.append([ int(m) for m in lines_p[x][i] if m!=','])
#         # regions.append(region)
#         print(regions)
#         print('aa')
    # regions.append(int(i) for i in lines[x] )
    # norm_coeff.append(int(i) for i in lines[x+1] if x<5)
    # norm_coeff.append(float(i) for i in lines[x+1] if x>=5)
# --- apply model to test data ---
f.close()
output_cls, output_reg = model.predict(x_test_1, verbose=1)
# print(output_cls[0,:,:,1])
# print(1)
'''
# NOTE: to review and test
# remove border pixels
border_padding = np.int((anchor_dims[-1][0] / anchor_stride) / 2) + 1

output_cls[0:border_padding, :, :] = 0
output_cls[output_cls.shape[0] - border_padding:, :, :] = 0
output_cls[:, 0:border_padding, :] = 0
output_cls[:, output_cls.shape[1] - border_padding:, :] = 0

output_reg[0:border_padding, :, :] = 0
output_reg[output_reg.shape[0] - border_padding:, :, :] = 0
output_reg[:, 0:border_padding, :] = 0
output_reg[:, output_reg.shape[1] - border_padding:, :] = 0
'''


# plot histogram of output bounding box sizes
# Y_test_pred_flat = output_cls.flatten()
#
# plt.hist(Y_test_pred_flat, density=False, bins=100)  # density=False shows counts, True shows density
# plt.axvline(0.5, color='k', linestyle='dashed', linewidth=1)
# plt.ylabel('Count')
# plt.xlabel('Probability values')
# plt.show()


if flag_save_intermediate_output:

    # --- create folders ---
    intermed_out_path = os.path.join(resultsPath, 'intermediate_output')
    if not os.path.exists(intermed_out_path):
        os.mkdir(intermed_out_path)

    # raw probability maps, 64-bit precision
    intermed_cls_out_path = os.path.join(resultsPath, 'intermediate_output', 'probability_maps')
    if not os.path.exists(intermed_cls_out_path):
        os.mkdir(intermed_cls_out_path)

    # raw regressor data, 64-bit precision
    intermed_reg_out_path = os.path.join(resultsPath, 'intermediate_output', 'regression')
    if not os.path.exists(intermed_reg_out_path):
        os.mkdir(intermed_reg_out_path)

    # --- save probability maps ---
    output_cls_pos = output_cls[:, :, :, :-1]   # num_images, h, w, channels

    for im_ind, output_cls_sample in enumerate(output_cls_pos):
        for channel in range(output_cls_sample.shape[-1]):
            np.savetxt(os.path.join(intermed_cls_out_path, str(im_ind).zfill(6) + '_' + str(channel) + '.txt'), output_cls_sample[:, :, channel], delimiter=',')

    # --- bounding box adjustments ---
    output_reg_pos = output_reg[:, :, :, :-1]  # num_images, h, w, channels

    for im_ind, output_reg_sample in enumerate(output_reg_pos):
        for channel in range(output_reg_sample.shape[-1]):    # 4 channels: delta_r, delta_c, h_percent, w_percent
            np.savetxt(os.path.join(intermed_reg_out_path, str(im_ind).zfill(6) + '_' + str(channel) + '.txt'), output_reg_sample[:, :, channel], delimiter=',')

    # --- normalization coefficients as float ---
    #NOTE: da se menuva so novi koeficienti

    # f = open(os.path.join(intermed_out_path, 'norm_coef_position.txt'), 'w')
    # f.write(str(reg_norm_coef_position))
    # f.close()
    #
    # f = open(os.path.join(intermed_out_path, 'norm_coef_size.txt'), 'w')
    # f.write(str(reg_norm_coef_size))
    # f.close()


# --- plot ground truth network output ---
prob_thr = 0.5
thr_clustering = 0.3
color_small = (0, 255, 0)
color_large = (0, 0, 255)
colors_list = (color_small, color_large)
# output_branch = 'regressor'
output_branch = 'classifier'
results_kamioni=os.path.join(dstResultsPath,version)
if not os.path.exists(os.path.join(results_kamioni,'kamioni')):
    os.mkdir(os.path.join(results_kamioni, 'kamioni'))
results_kamioni1 = os.path.join(results_kamioni, 'kamioni')
anchorless_genertor_plot.save_results_anchorless_limits_normalizedByRegion(resultsPath, results_path_nms, results_kamioni1, x_test, output_cls, output_reg, anchor_stride, prob_thr,
                                                                           regions,norm_coeff,dest_matrix, thr_clustering, colors_list,flag_save_coords=False)
