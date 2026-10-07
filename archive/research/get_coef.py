import os
import pickle
# src=r'D:\Monika\Models\ModelsTestingNew3\Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor_Renamed_PRVVTORsloj_16filters_BEZ_SKIP_1INC\reg_coef'
src=r'D:\Monika\Models\Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor1_All_Negatives_Renamed_alpha_1_reg_0.01_DIOU_LOSS_1_novi_fp_2_Focal_CATEGORICAL_norm_reg1\reg_coef'

for filename in os.listdir(src):
    print(filename)
    fid1_2 = open(os.path.join(src, filename), 'rb')
    reg_norm_coef_row = pickle.load(fid1_2)

    print(reg_norm_coef_row)
