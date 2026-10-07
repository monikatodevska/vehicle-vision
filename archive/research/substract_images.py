import numpy as np
import cv2
elena=r'D:\Monika\RamkiOdElena\Frames\111.bmp'
monika=r'D:\Monika\Results\kam25\ZaTest\luge_na_pat_5fps_trim1\image_000115.bmp'
slikaElena=cv2.imread(elena, 0)
slikaMonika=cv2.imread(monika, 0)

cv2.imshow('s', slikaElena)
cv2.waitKey(0)
cv2.imshow('s2', slikaMonika)
cv2.waitKey(0)
print(np.sum(np.abs(np.array(slikaElena)-np.array(slikaMonika))))
razlika=np.abs(np.array(slikaElena)-np.array(slikaMonika))

cv2.imshow('slika', razlika)
cv2.waitKey(0)
