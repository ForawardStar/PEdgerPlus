import caffe
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
import cv2

# 设置模式
caffe.set_mode_cpu()

# 加载模型
net = caffe.Net('/home/fyb/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence_4recurrentFast_noMS_LargerModel_tune08_ReduceSize_Testset3/EdgeRecurrent_NoInterp.prototxt', '/home/fyb/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence_4recurrentFast_noMS_LargerModel_tune08_ReduceSize_Testset3/EdgeRecurrent_NoInterp.caffemodel', caffe.TEST)

# 加载图像
image = caffe.io.load_image('input.png')

# 预处理图像
transformer = caffe.io.Transformer({'data': net.blobs['blob1'].data.shape})
transformer.set_transpose('data', (2,0,1))
#transformer.set_mean('data', np.load('caffe_ilsvrc12/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))
net.blobs['blob1'].data[...] = transformer.preprocess('data', image)

# 运行网络
net.forward()

# 获取特征图
feature_maps = net.blobs['conv_blob1'].data
print("feature_maps type:{}  ,  feature_maps shape:{}".format(type(feature_maps), feature_maps.shape))


# 显示结果
cv2.imwrite('edges.png', feature_maps[0:1, 0:3, :, :])

