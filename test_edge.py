import caffe
import numpy as np
from skimage import feature
import cv2
import os
import time

def sigmoid(x):
    return np.exp(x - 0.5) / (np.exp(x - 0.5) + np.exp(0.5 - x))

# Running Mode
caffe.set_mode_cpu()

# Loading Pre-trained Checkpoints
net1 = caffe.Net('models/EdgeModel1.prototxt', 'models/EdgeModel.caffemodel', caffe.TEST)

net2 = caffe.Net('models/EdgeModel2.prototxt', 'models/EdgeModel.caffemodel', caffe.TEST)


print("type:{}  ,  shape:{}".format(type(net1.blobs['blob1'].data), net1.blobs['blob1'].data.shape))
# Loading Images
input_path = "data/HED-BSDS/test/"
save_path = "edge_results/"
os.makedirs(save_path, exist_ok=True)
filenames = os.listdir(input_path)
for filename in filenames:
    image = caffe.io.load_image(input_path + filename)
    if image.shape[0] == 481:
        # Pre-processing images
        transformer = caffe.io.Transformer({'data': net1.blobs['blob1'].data.shape})
        transformer.set_transpose('data', (2,0,1)) # Converting HWC to CHW
        net1.blobs['blob1'].data[...] = transformer.preprocess('data', image)

        # network inference
        net1.forward()

        # Generating results
        features = net1.blobs['conv_blob119'].data.squeeze()
        res = sigmoid(features)
        print("image shape:{}  ,  features shape:{}".format(image.shape, features.shape))

        # save results
        cv2.imwrite(save_path + filename, res*255)
    else:
        # Pre-processing images
        transformer = caffe.io.Transformer({'data': net2.blobs['blob1'].data.shape})
        transformer.set_transpose('data', (2,0,1)) # Convert HWC to CHW
        net2.blobs['blob1'].data[...] = transformer.preprocess('data', image)

        # network inference
        net2.forward()

        # Generating results
        features = net2.blobs['conv_blob119'].data.squeeze()
        res = sigmoid(features)
        print("image shape:{}  ,  features shape:{}".format(image.shape, features.shape))

        # save results
        cv2.imwrite(save_path + filename, res*255)






