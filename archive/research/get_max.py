import os
import pickle
import numpy as np
import tqdm
file_path=r'M:\Monika\Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor1_All_Negatives_Renamed\reg_coef_latest'
groundtruthfiles=r'M:\Monika\Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor1_All_Negatives_Renamed\GT'
lista=os.listdir(groundtruthfiles)
out_class_list = []
out_reg_list = []
i=0
for filename in lista:
    # filename = 'image' + str(id).zfill(6) + '.txt'
    fid1 = open(os.path.join(groundtruthfiles, filename), 'rb')
    out_class_dims = pickle.load(fid1)
    out_class_back = pickle.load(fid1)
    out_reg_dims = pickle.load(fid1)
    out_reg_back = pickle.load(fid1)
    fid1.close()
    # out_class = np.reshape(out_class_back, (out_class_dims[0], out_class_dims[1], out_class_dims[2]))
    # out_class_list.append(out_class)
    out_reg = np.reshape(out_reg_back, (out_reg_dims[0], out_reg_dims[1], out_reg_dims[2]))
    out_reg_list.append(out_reg)
    print(i)
    i+=1
# out_class_list = np.array(out_class_list)
out_reg_list = np.array(out_reg_list)

reg_norm_coef_position_rows=np.max(np.abs(out_reg_list[:, :, :, 0]))

fid1 = open(os.path.join(file_path, 'reg_norm_coef_position_rows.txt'), 'wb+')
pickle.dump(reg_norm_coef_position_rows,fid1)
fid1.close()
# out_reg_list = out_reg_list / reg_norm_coef



reg_norm_coef_position_cols=np.max(np.abs(out_reg_list[:, :, :, 1]))

fid1 = open(os.path.join(file_path, 'reg_norm_coef_position_cols.txt'), 'wb+')
pickle.dump(reg_norm_coef_position_cols,fid1)
fid1.close()



reg_norm_coef_size_height=np.max(np.abs(out_reg_list[:, :, :, 2]))

fid1 = open(os.path.join(file_path, 'reg_norm_coef_size_height.txt'), 'wb+')
pickle.dump(reg_norm_coef_size_height,fid1)
fid1.close()


reg_norm_coef_size_width=np.max(np.abs(out_reg_list[:, :, :, 3]))

fid1 = open(os.path.join(file_path, 'reg_norm_coef_size_width.txt'), 'wb+')
pickle.dump(reg_norm_coef_size_width,fid1)
fid1.close()





