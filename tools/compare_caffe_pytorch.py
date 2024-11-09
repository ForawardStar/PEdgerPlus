import os
import cv2
import numpy as np
import math

input_path1 = "/home/fyb/PEdgerPlus/edges5/"
input_path2 = "/home/fyb/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence_4recurrentFast_noMS_LargerModel_tune08_ReduceSize_Testset3/TestCaffe_ckt_0012/"
filenames = os.listdir(input_path1)
count = 0
total_diff = 0
for filename in filenames:
    img1 = cv2.imread(input_path1 + filename).astype(np.float64)
    img2 = cv2.imread(input_path2 + filename.replace(".jpg", ".png")).astype(np.float64)

    diff = np.mean(np.abs(img1 - img2))
    print("img1:{}  ,  img2:{}  ,  diff:{}".format(np.mean(img1), np.mean(img2), diff))
    count = count + 1
    total_diff = total_diff + diff
print(total_diff / count)

