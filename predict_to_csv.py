import os
import cv2
import csv
import numpy as np
from PIL import Image

from unet import Unet

if __name__ == "__main__":
    unet = Unet()

    txt_lines = None
    all_lines = list()
    new_data = list()
    with open("input.txt", "r") as f:
        txt_lines = f.readlines()
    for one_line in txt_lines:
        all_lines.append(one_line.replace("\n", ""))

    for one in all_lines:
        one_res = list()
        image_path = one
        one_res.append(image_path)
        image = Image.open(image_path)
        r_image = unet.detect_image(image)
        image_area = r_image.shape[0] * r_image.shape[1]
        pred_ratio = len(np.where(r_image==1)[0]) / image_area
        print("pred_ratio:   ", pred_ratio)
        one_res.append(pred_ratio)
        new_data.append(one_res)

    newFile = open("output.csv", "w", newline="")
    writer = csv.writer(newFile)
    # writer.writerow(["ID", "Label"])
    writer.writerows(new_data)
    newFile.close()