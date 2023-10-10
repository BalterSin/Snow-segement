import os
import cv2
import numpy as np
from PIL import Image

from unet import Unet

if __name__ == "__main__":
    unet = Unet()

    dir_origin_path = "VOCdevkit/VOC2007/JPEGImages/"
    dir_save_path   = "result_images/"

    image_name = "02621.jpg"

    image_path = dir_origin_path + image_name
    image = Image.open(image_path)
    r_image = unet.detect_image(image)
    image_area = r_image.shape[0] * r_image.shape[1]
    pred_ratio = len(np.where(r_image==1)[0]) / image_area * 100
    print("pred_ratio:   ", pred_ratio)

    r_image = r_image.astype(np.uint8)
    dst = Image.fromarray(r_image, 'P')
    bin_colormap = [0, 0, 0] + [255, 255, 255] * 254
    dst.putpalette(bin_colormap)
    dst.save(dir_save_path + image_name.replace(".jpg", ".png"))

