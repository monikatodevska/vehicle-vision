import matplotlib.pyplot as plt
import numpy as np
import os

path1=r'D:\Monika\Models\Training_PEDESTRIANS_HalfHD_New5_All_Negatives_Renamed_gamma5_1'
path2=r'D:\Monika\Models\Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor1_All_Negatives_Renamed_alpha_1_reg_0.01_2'
dst=os.path.join(path2,'grafici')
if not os.path.exists(dst):
    os.makedirs(dst,exist_ok=True)

losses=[]
losses_2=[]
lines = []
losses_val=[]
losses_2_val=[]
flag_bagnat=False
reg=False
losses_reg_train=[]
losses_reg_val=[]
with open(os.path.join(path1,'train_class.txt')) as f:
    lines = f.readlines()
for line in lines:
    losses.append(line.replace("\n", ""))

with open(os.path.join(path1, 'val_class.txt')) as f:
        lines = f.readlines()
for line in lines:
    losses_val.append(line.replace("\n", ""))
    # line.strip('\n')

# exit(1)
losses = [np.float32(i) for i in losses]
losses_val = [np.float32(i) for i in losses_val]


if reg:
    with open(os.path.join(path1,'train_reg.txt')) as f:
        lines = f.readlines()
    for line in lines:
        losses_reg_train.append(line.replace("\n", ""))

    with open(os.path.join(path1, 'train_val.txt')) as f:
            lines = f.readlines()
    for line in lines:
        losses_reg_val.append(line.replace("\n", ""))
    losses_reg_train = [np.float32(i) for i in losses_reg_train]
    losses_reg_val = [np.float32(i) for i in losses_reg_val]

if flag_bagnat:
    with open(os.path.join(path2,'train_reg.txt')) as f:
        lines = f.readlines()
    for line in lines:
        losses_2.append(line.replace("\n", ""))

    losses_2=[np.float32(i) for i in losses_2]

    losses.extend(losses_2)


    with open(os.path.join(path2,'train_val.txt')) as f:
        lines = f.readlines()
    for line in lines:
        losses_2_val.append(line.replace("\n", ""))

    losses_2_val=[np.float32(i) for i in losses_2_val]

    losses_val.extend(losses_2_val)

plt.figure('1')
plt.plot(range(0,len(losses)), losses,'r')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(0,len(losses_val)),losses_val,'b')
plt.title('Classification Loss')
plt.grid()
# plt.savefig(os.path.join(dst,'classification_loss.png'))
# plt.show() #clears everything, ako e pred savefig bela slika se zacuvuva ??????????????

plt.figure('2')

plt.plot(range(0,len(losses_reg_train)), losses_reg_train,'r')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Regression Loss')
plt.plot(range(0,len(losses_reg_val)),losses_reg_val,'b')
plt.grid()
# plt.savefig(os.path.join(dst,'classification_loss.png'))
# plt.show() #clears everything, ako e pred savefig bela slika se zacuvuva ??????????????

sum_losses_train=[sum(x) for x in zip(losses, losses_reg_train)]
sum_losses_val=[sum(x) for x in zip(losses_val, losses_reg_val)]

plt.figure('3')
plt.plot(range(0,len(sum_losses_train)), sum_losses_train,'r')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Joint Loss')
plt.plot(range(0,len(sum_losses_val)),sum_losses_val,'b')
plt.grid()
# plt.savefig(os.path.join(dst,'classification_loss.png'))
plt.show() #clears everything, ako e pred savefig bela slika se zacuvuva ??????????????

print(1)

# print(losses)

