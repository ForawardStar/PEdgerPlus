import os
import cv2
import numpy as np
import math

ori_path = "/home/fyb/PEdgerPlus/edges5/"
save_path = "/home/fyb/PEdgerPlus/edges5_png/"
os.makedirs(save_path, exist_ok=True)
filenames = os.listdir(ori_path)
for filename in filenames:
    img = cv2.imread(ori_path + filename).astype(np.float64)
    cv2.imwrite(save_path + filename.replace(".jpg", ".png"), img)


