import caffe  
  
# 设置 Caffe 模式为 GPU（如果可用）  
caffe.set_mode_gpu()  
  
# 加载训练好的模型  
model_def = 'path/to/train_val.prototxt'  
model_weights = 'path/to/output/lenet_iter_100000.caffemodel'  
  
net = caffe.Net(model_def, model_weights, caffe.TEST)  
  
# 加载测试数据（假设测试数据也是 LMDB 格式）  
test_data_layer = 'data'  
test_data_path = 'path/to/test_data_lmdb'  
  
net.blobs[test_data_layer].data[...] = caffe.io.load_lmdb_data(test_data_path)['data']  
net.blobs[test_data_layer].reshape(len(net.blobs[test_data_layer].data), *net.blobs[test_data_layer].data.shape[1:])  
  
# 前向传播  
output = net.forward()  
  
# 获取准确率  
accuracy = output['accuracy']  
print('Test Accuracy: {:.2f}%'.format(accuracy * 100))
