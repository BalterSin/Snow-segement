import os
import cv2
import numpy as np
from PIL import Image

from unet import Unet

if __name__ == "__main__":
    unet = Unet()

    dir_origin_path = "VOCdevkit/VOC2007/JPEGImages/"
    dir_mask_path = "VOCdevkit/VOC2007/SegmentationClass/"
    dir_save_path   = "test_results/"

    all_images = os.listdir(dir_origin_path)

    all_num = len(all_images)
    correct_list = list()

    # for one_image in all_images:
    for index in range(all_num):
        one_image = all_images[index]
        print("-------------------------" + str(index+1) + "/" + str(all_num) + "-------------------------")
        print(one_image)
        one_image_path = dir_origin_path + one_image
        image = Image.open(one_image_path)
        r_image = unet.detect_image(image)
        image_area = r_image.shape[0] * r_image.shape[1]
        pred_ratio = len(np.where(r_image==1)[0]) / image_area * 100
        mask_img = Image.open(dir_mask_path + one_image.replace(".jpg", ".png"))
        mask_array = np.array((mask_img))
        mask_ratio = len(np.where(mask_array==1)[0]) / image_area * 100
        print("mask_ratio:   ", mask_ratio)
        print("pred_ratio:   ", pred_ratio)
        if pred_ratio >= (mask_ratio - 20) and pred_ratio <= (mask_ratio + 20):
            print(True)
            correct_list.append(1)
        else:
            print(False)

        r_image = r_image.astype(np.uint8)
        dst = Image.fromarray(r_image, 'P')
        bin_colormap = [0, 0, 0] + [255, 255, 255] * 254
        dst.putpalette(bin_colormap)
        dst.save(dir_save_path + one_image.replace(".jpg", ".png"))

    print("Accuracy:   ", len(correct_list) / all_num * 100)
