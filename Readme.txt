This is a road snow coverage estimator.
The environment and dependent libraries are documented in "requirement.txt".
The model has been trained and the appropriate weights have been loaded.
If you want to reproduce the training process, run "train.py". Note that the pre-training weights for "model_data/unet_resnet_voc.pth" are preset in "train.py"
If you want to make an estimate of snow cover for images, here I have designed three ways.

----------------------------predict_to_csv.py---------------------------------------

Enter the image path you want to estimate line by line in "input.txt", and run "predict_to_csv.py", the estimation result will be output as "output.csv"

----------------------------predict_all_imgs.py---------------------------------------

Run this file and you will see the estimation results for all images and get an accuracy rate.

----------------------------predict_one_img.py---------------------------------------

Run this file and you will get the estimation result for a particular image. You can change the image_name to modify the input image.

Thank you for reading!