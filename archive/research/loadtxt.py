reg_coef=r'D:\Monika\Models\v3200_0.6_0.7_normall_with_translations_up_down-with_inceptions_and_skip_softmax_CUSTOM_CATEGORICAL_LOSS_AXES__10^-1_reg_coef_FINETUNE_CEL\reg_coef'


pos_col='reg_norm_coef_position_cols.txt'
pos_rows='reg_norm_coef_position_rows.txt'
size_height='reg_norm_coef_size_height.txt'
size_width='reg_norm_coef_size_width.txt'



import os
import pickle

fid=open(os.path.join(reg_coef,pos_col), 'rb')
pos_c=pickle.load(fid)
fid.close()

fid=open(os.path.join(reg_coef,pos_rows), 'rb')
pos_r=pickle.load(fid)
fid.close()

fid=open(os.path.join(reg_coef,size_height), 'rb')
size_h=pickle.load(fid)
fid.close()

fid=open(os.path.join(reg_coef,size_width), 'rb')
size_w=pickle.load(fid)
fid.close()



print(pos_c)
print(pos_r)
print(size_h)
print(size_w)