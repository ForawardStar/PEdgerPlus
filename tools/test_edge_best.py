import caffe
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
import cv2
import os
import time

def sigmoid(x):
    return np.exp(x - 0.5) / (np.exp(x - 0.5) + np.exp(0.5 - x))

# 设置模式
caffe.set_mode_cpu()

# 加载模型
net1 = caffe.Net('/home/fyb/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence_4recurrentFast_noMS_LargerModel_tune08_ReduceSize_Testset3/EdgeRecurrent_NoInterp.prototxt', '/home/fyb/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence_4recurrentFast_noMS_LargerModel_tune08_ReduceSize_Testset3/EdgeRecurrent_NoInterp.caffemodel', caffe.TEST)

net2 = caffe.Net('/home/fyb/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence_4recurrentFast_noMS_LargerModel_tune08_ReduceSize_Testset3/EdgeRecurrent_NoInterp2.prototxt', '/home/fyb/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence_4recurrentFast_noMS_LargerModel_tune08_ReduceSize_Testset3/EdgeRecurrent_NoInterp.caffemodel', caffe.TEST)


print("type:{}  ,  shape:{}".format(type(net1.blobs['blob1'].data), net1.blobs['blob1'].data.shape))
# 加载图像
input_path = "/home/fyb/HED-BSDS/test/"
save_path = "/home/fyb/PEdgerPlus/edges5/"
os.makedirs(save_path, exist_ok=True)
filenames = os.listdir(input_path)
for filename in filenames:
    image = caffe.io.load_image(input_path + filename)
    if image.shape[0] == 481:
        # 预处理图像
        transformer = caffe.io.Transformer({'data': net1.blobs['blob1'].data.shape})
        transformer.set_transpose('data', (2,0,1)) # 从HWC转换到CHW
        net1.blobs['blob1'].data[...] = transformer.preprocess('data', image)

        # 运行网络
        net1.forward()

        # 获取特征图
        features = net1.blobs['conv_blob85'].data.squeeze()
        res = sigmoid(features)
        print("image shape:{}  ,  features shape:{}".format(image.shape, features.shape))

        # 显示结果
        cv2.imwrite(save_path + filename, res*255)
    else:
        # 预处理图像
        transformer = caffe.io.Transformer({'data': net2.blobs['blob1'].data.shape})
        transformer.set_transpose('data', (2,0,1)) # 从HWC转换到CHW
        net2.blobs['blob1'].data[...] = transformer.preprocess('data', image)

        # 运行网络
        net2.forward()

        # 获取特征图
        features = net2.blobs['conv_blob85'].data.squeeze()
        res = sigmoid(features)
        print("image shape:{}  ,  features shape:{}".format(image.shape, features.shape))

        # 显示结果
        cv2.imwrite(save_path + filename, res*255)






