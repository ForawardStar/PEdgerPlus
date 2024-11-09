import caffe
import numpy as np
#import matplotlib.pyplot as plt
from skimage import feature
import cv2
import os
import time

def sigmoid(x):
    return np.exp(x - 0.5) / (np.exp(x - 0.5) + np.exp(0.5 - x))

# 设置模式
caffe.set_mode_cpu()

# 加载模型
net = caffe.Net('/home/li/fyb/Caffe-anzhuang/caffe-master2/python/examples/bsds500/test.prototxt', '/home/li/fyb/Caffe-anzhuang/caffe-master2/python/examples/bsds500/snapshot_iter_9500.caffemodel', caffe.TEST)

print("type:{}  ,  shape:{}".format(type(net.blobs['data'].data), net.blobs['data'].data.shape))
# 加载图像
save_path = "/home/li/fyb/Caffe-anzhuang/caffe-master2/python/edges_caffetraining/"
os.makedirs(save_path, exist_ok=True)

for ii in range(200):
    # 预处理图像
    # 运行网络
    net.forward()

    # 获取特征图
    features = net.blobs['conv4'].data.squeeze()
    res = sigmoid(features)
    print("res shape:{}  ,  features shape:{}".format(res.shape, features.shape))

    # 显示结果
    cv2.imwrite(save_path + str(ii) + ".png", res*255)
    
