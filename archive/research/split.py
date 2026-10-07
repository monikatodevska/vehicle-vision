import os
import numpy as np

import shutil
srcImagesPath = r'E:\Science\Monika\site'

allFileNames=os.listdir(srcImagesPath)
train_ratio=0.7
val_ratio=0.15
test_ratio=0.15
train_FileNames, val_FileNames, test_FileNames=np.split(np.array(allFileNames), [int(len(allFileNames)*0.7), int(len(allFileNames)*0.85)])
train_FileNames=[srcImagesPath + '/' + name for name in train_FileNames.tolist()]
val_FileNames = [srcImagesPath + '/' + name for name in val_FileNames.tolist()]
test_FileNames = [srcImagesPath + '/' + name for name in test_FileNames.tolist()]
for name in train_FileNames:
    shutil.copy(name, r'E:\Science\Monika\train1')

for name in val_FileNames:
    shutil.copy(name, r'E:\Science\Monika\val1')


for name in test_FileNames:
    shutil.copy(name, r'E:\Science\Monika\test1')


