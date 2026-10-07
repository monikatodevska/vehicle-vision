from PIL import Image, ImageEnhance
import os
import cv2
import numpy as np
'''
factor -to brighten or darken the image
factor: 1-original image
factor < 1 darkened image
factor>1 brightened image
'''
probni_sliki=r'\\192.168.1.153\d\KlasifikacijaVozila\proba-sliki\miladinovci_soncevo'
zatemneti_brightness=r'\\192.168.1.153\d\KlasifikacijaVozila\brightness_reduced40'
zatemneti_contrast=r'\\192.168.1.153\d\KlasifikacijaVozila\contrast_reduced40'

#
# if not os.path.exists(zatemneti):
#         os.mkdir(os.path.join(zatemneti))
def change_brightness(image, factor):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(im)

    # image brightness enhancer
    enhancer = ImageEnhance.Brightness(im)

    # darkens the image

    im_output = enhancer.enhance(factor)
    im_output=np.array(im_output)
    im_output=cv2.cvtColor(im_output, cv2.COLOR_RGB2BGR)

    return im_output


def change_contrast(image, factor_contrast):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    im = Image.fromarray(im)
    # image brightness enhancer
    enhancer = ImageEnhance.Contrast(im)

    im_output = enhancer.enhance(factor_contrast)
    im_output=np.array(im_output)
    im_output=cv2.cvtColor(im_output, cv2.COLOR_RGB2BGR)
    return im_output



for im_name in os.listdir(probni_sliki):
    image=cv2.imread(os.path.join(probni_sliki,im_name))
    im_output=change_brightness(image, 0.6)
    cv2.imwrite(os.path.join(zatemneti_brightness,im_name),im_output)
    im_output1=change_contrast(image,0.6)
    cv2.imwrite(os.path.join(zatemneti_contrast,im_name[:-4]+'contrast'+'.bmp'),im_output1)


    # cv2.imshow("slika_orig",image)
    # cv2.waitKey(0)
    # cv2.imshow("bright_dec",im_output)
    # cv2.waitKey(0)
#
#     im_output = change_brightness(image, 1.1)
#     cv2.imshow("slika_orig", image)
#     cv2.waitKey(0)
#     cv2.imshow("bright_inc", im_output)
#     cv2.waitKey(0)
#
#     im_output = change_contrast(image, 0.8)
#     cv2.imshow("slika_orig", image)
#     cv2.waitKey(0)
#     cv2.imshow("kon_dec", im_output)
#     cv2.waitKey(0)
#
#     im_output=change_contrast(image, 1.1)
#     cv2.imshow("slika_orig", image)
#     cv2.waitKey(0)
#     cv2.imshow("kontrast_inc",im_output)
#     cv2.waitKey(0)




