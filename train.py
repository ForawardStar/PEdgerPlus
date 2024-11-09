import os
import cv2
import csv
import glob
import numpy
import random
import argparse
import lmdb
# To silence Caffe! Must be added before importing Caffe or modules which
# are importing Caffe.
os.environ['GLOG_minloglevel'] = '3'
import caffe
import torch
import tools.solvers
import tools.lmdb_io
import tools.prototxt
import tools.pre_processing

caffe.set_mode_cpu()

def get_parser():
    """
    Get the parser.
    
    :return: parser
    :rtype: argparse.ArgumentParser
    """
    
    parser = argparse.ArgumentParser(description = 'Deep learning for edge detection on BSDS500.')
    parser.add_argument('--mode', default = 'train',
                        help = 'Mode to run: "extract", "subsample_test" or "train"')
    parser.add_argument('--working_directory', default = 'examples/bsds500', type = str,
                       help = 'path to the working directory, see documentation of this example')
    parser.add_argument('--train_lmdb', default = 'examples/bsds500/test_lmdb', type = str,
                       help = 'path to train LMDB')
    parser.add_argument('--train_gt_lmdb', default = 'examples/bsds500/test_gt_lmdb', type = str,
                       help = 'path to ground truth LMDB')
    parser.add_argument('--test_lmdb', default = 'examples/bsds500/test_lmdb', type = str,
                       help = 'path to test LMDB')
    parser.add_argument('--test_gt_lmdb', default = 'examples/bsds500/test_gt_lmdb', type = str,
                       help = 'path to test ground truth LMDB')
    parser.add_argument('--iterations', dest = 'iterations', type = int,
                        help = 'number of iterations to train or resume',
                        default = 10000)
                        
    return parser
   
def csv_read(csv_file, delimiter = ','):
    """
    Read a CSV file into a numpy.ndarray assuming that each row has the same
    number as columns.
    
    :param csv_file: path to CSV file
    :type csv_file: string
    :param delimiter: delimiter between cells
    :type delimiter: string
    :return: CSV contents as Numpy array as float
    :rtype: numpy.ndarray
    """
    
    cols = -1
    array = []
    
    with open(csv_file) as f:
        for cells in csv.reader(f, delimiter = delimiter):
                cells = [cell.strip() for cell in cells if len(cell.strip()) > 0]
                
                if len(cells) > 0:
                    if cols < 0:
                        cols = len(cells)
                    
                    assert cols == len(cells), "CSV file does not contain a consistent number of columns"
                    
                    cells = [float(cell) for cell in cells]
                    array.append(cells)
    
    return numpy.array(array)   
   
def main_extract():
    """
    Extracts train and test samples from the train and test images and ground truth
    in bsds500/csv_groundTruth and bsds500/images. For each positive edge pixels,
    a quadratic patch is extracted. For non-edge pixels, all patches are subsampled
    by only taking 20% of the patches.
    
    It might be beneficial to also run :func:`examples.bsds500.main_subsample_test`
    on the extracted test LMDB for efficient testing during training.
    """
    
    def extract(directory, lmdb_path):
        assert not os.path.exists(lmdb_path), "%s already exists" % lmdb_path

        segmentation_files = [filename for filename in os.listdir(args.working_directory + '/csv_groundTruth/' + directory) if filename[-4:] == '.csv']
        
        lmdb_path = args.working_directory + '/' + directory + '_lmdb'
        lmdb = tools.lmdb_io.LMDB(lmdb_path)
                    
        s = 1
        for segmentation_file in segmentation_files:
            image_file = args.working_directory + '/images/' + directory + '/' + segmentation_file[:-6] + '.jpg'
            image = cv2.imread(image_file)
            segmentation = csv_read(args.working_directory + '/csv_groundTruth/' + directory + '/' + segmentation_file)
            
            inner = segmentation[1:segmentation.shape[0] - 2, 1:segmentation.shape[1] - 2]
            inner_top = segmentation[0:segmentation.shape[0] - 3, 1:segmentation.shape[1] - 2]
            inner_left = segmentation[1:segmentation.shape[0] - 2, 0:segmentation.shape[1] - 3]
            
            segmentation[1:segmentation.shape[0] - 2, 1:segmentation.shape[1] - 2] = numpy.abs(inner - inner_top) + numpy.abs(inner - inner_left)
            
            segmentation[:, :2] = 0
            segmentation[:, segmentation.shape[1] - 3:] = 0
            segmentation[:2, :] = 0
            segmentation[segmentation.shape[0] - 3:, :] = 0
            
            segmentation[segmentation > 0] = 1
            
            images = []
            labels = []
            
            k = 3
            n = 0
            for i in range(k, segmentation.shape[0] - k):
                for j in range(k, segmentation.shape[1] - k):
                    
                    r = random.random()
                    patch = image[i - k:i + k + 1, j - k:j + k + 1, :]
                    
                    if segmentation[i, j] > 0:
                        images.append(patch)
                        labels.append(1)
                    elif r > 0.8:
                        images.append(patch)
                        labels.append(0)
                    
                    n += 1
            
            lmdb.write(images, labels)
            print(str(s) + '/' + str(len(segmentation_files)))
            s += 1
    
    extract('train', args.train_lmdb)
    extract('val', args.test_lmdb)
    
def main_subsample_test():
    """
    Subsample the test LMDB by only taking 5% of the samples. The original test
    LMDB is renamed by appending '_full' and a newtest  is created having the same
    name as the original one.
    """
    
    test_in_lmdb = args.test_lmdb + '_full'
    test_out_lmdb = args.test_lmdb
    
    assert os.path.exists(test_out_lmdb), "LMDB %s not found" % test_out_lmdb
    os.rename(test_out_lmdb, test_in_lmdb)
    
    pp_in = tools.pre_processing.PreProcessingInputLMDB(test_in_lmdb)
    pp_out = tools.pre_processing.PreProcessingOutputLMDB(test_out_lmdb)
    pp = tools.pre_processing.PreProcessingSubsample(pp_in, pp_out, 0.05)
    pp.run()    
    
def main_train():
    """
    After running :func:`examples.bsds500.main_train`, a network can be trained.
    """
    def residual_block(net, num_output, dilation=1, prefix="res"):
        # 基础卷积层
        padding = (3 - 1) // 2 + dilation - 1
        res1 = caffe.layers.Convolution(net, num_output=num_output, kernel_size=3, dilation=dilation, pad=padding,
                          weight_filler=dict(type='constant', value=0), bias_filler=dict(type='constant', value=0),
                          param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
        relu_res1 = caffe.layers.ReLU(res1, in_place = True)
        res2 = caffe.layers.Convolution(relu_res1, num_output=num_output, kernel_size=3, dilation=dilation, pad=padding,
                          weight_filler=dict(type='constant', value=0), bias_filler=dict(type='constant', value=0),
                          param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

        # 形成残差连接
        return caffe.layers.Eltwise(res2, net, operation=caffe.params.Eltwise.SUM)

    
    def network(train_lmdb_path, gt_lmdb_path, batch_size):
        """
        The network definition given the LMDB path and the used batch size.
        
        :param lmdb_path: path to LMDB to use (train or test LMDB)
        :type lmdb_path: string
        :param batch_size: batch size to use
        :type batch_size: int
        :return: the network definition as string to write to the prototxt file
        :rtype: string
        """
        
        net = caffe.NetSpec()
            
        net.data, _ = caffe.layers.Data(batch_size = batch_size, 
                                                 backend = caffe.params.Data.LMDB, 
                                                 source = train_lmdb_path, 
                                                 transform_param = dict(scale=1/255.0), 
                                                 ntop = 2)

        net.labels, _ = caffe.layers.Data(batch_size = batch_size,
                                                 backend = caffe.params.Data.LMDB,
                                                 source = gt_lmdb_path,
                                                 transform_param = dict(scale=1/255.0),
                                                 ntop = 2)
        
        # Encoding Module
        net.head1 = caffe.layers.Convolution(net.data, kernel_size = 3, pad=1, num_output = 16, 
                                             weight_filler = dict(type = 'constant', value=0))
        net.head_relu1 = caffe.layers.ReLU(net.head1, in_place = True)
        net.head2 = residual_block(net.head_relu1, num_output=16, prefix='head2')
        net.head_relu2 = caffe.layers.ReLU(net.head2, in_place = True)
        net.head3 = residual_block(net.head_relu2, num_output=16, prefix='head3')
        net.head_relu3 = caffe.layers.ReLU(net.head3, in_place = True)


        # Stage 1
        net.stage1_conv1 = residual_block(net.head_relu3, num_output=16, prefix='stage1_conv1')
        net.stage1_relu1 = caffe.layers.ReLU(net.stage1_conv1, in_place = True)
        net.stage1_conv2 = residual_block(net.stage1_relu1, num_output=16, dilation=2, prefix='stage1_conv2')
        net.stage1_relu2 = caffe.layers.ReLU(net.stage1_conv2, in_place = True)
        net.stage1_conv3 = residual_block(net.stage1_relu2, num_output=16, dilation=4, prefix='stage1_conv3')
        net.stage1_relu3 = caffe.layers.ReLU(net.stage1_conv3, in_place = True)
        net.stage1_conv4 = residual_block(net.stage1_relu3, num_output=16, dilation=8, prefix='stage1_conv4')
        net.stage1_relu4 = caffe.layers.ReLU(net.stage1_conv4, in_place = True)
        net.stage1_pool = caffe.layers.Pooling(net.stage1_relu4, pool=caffe.params.Pooling.MAX, kernel_size=2, stride=2)

        # Stage 2
        net.stage2_conv1 = residual_block(net.stage1_relu4, num_output=16, prefix='stage2_conv1')
        net.stage2_relu1 = caffe.layers.ReLU(net.stage2_conv1, in_place = True)
        net.stage2_conv2 = residual_block(net.stage2_relu1, num_output=16, prefix='stage2_conv2')
        net.stage2_relu2 = caffe.layers.ReLU(net.stage2_conv2, in_place = True)
        net.stage2_conv3 = residual_block(net.stage2_relu2, num_output=16, dilation=2, prefix='stage2_conv3')
        net.stage2_relu3 = caffe.layers.ReLU(net.stage2_conv3, in_place = True)
        net.stage2_conv4 = residual_block(net.stage2_relu3, num_output=16, dilation=4, prefix='stage2_conv4')
        net.stage2_relu4 = caffe.layers.ReLU(net.stage2_conv4, in_place = True)
        net.stage2_conv5 = residual_block(net.stage2_relu4, num_output=16, dilation=8, prefix='stage2_conv5')
        net.stage2_relu5 = caffe.layers.ReLU(net.stage2_conv5, in_place = True)
        net.stage2_conv6 = residual_block(net.stage2_relu5, num_output=16, prefix='stage2_conv6')
        net.stage2_relu6 = caffe.layers.ReLU(net.stage2_conv6, in_place = True)
        net.stage2_pool = caffe.layers.Pooling(net.stage2_relu6, pool=caffe.params.Pooling.MAX, kernel_size=2, stride=2)
        

        # Stage 3
        net.stage3_conv1 = residual_block(net.stage2_relu6, num_output=16, prefix='stage3_conv1')
        net.stage3_relu1 = caffe.layers.ReLU(net.stage3_conv1, in_place = True)
        net.stage3_conv2 = residual_block(net.stage3_relu1, num_output=16, prefix='stage3_conv2')
        net.stage3_relu2 = caffe.layers.ReLU(net.stage3_conv2, in_place = True)
        net.stage3_conv3 = residual_block(net.stage3_relu2, num_output=16, prefix='stage3_conv3')
        net.stage3_relu3 = caffe.layers.ReLU(net.stage3_conv3, in_place = True)
        net.stage3_conv4 = residual_block(net.stage3_relu3, num_output=16, dilation=2, prefix='stage3_conv4')
        net.stage3_relu4 = caffe.layers.ReLU(net.stage3_conv4, in_place = True)
        net.stage3_conv5 = residual_block(net.stage3_relu4, num_output=16, dilation=2, prefix='stage3_conv5')
        net.stage3_relu5 = caffe.layers.ReLU(net.stage3_conv5, in_place = True)
        net.stage3_conv6 = residual_block(net.stage3_relu5, num_output=16, dilation=4, prefix='stage3_conv6')
        net.stage3_relu6 = caffe.layers.ReLU(net.stage3_conv6, in_place = True)
        net.stage3_conv7 = residual_block(net.stage3_relu6, num_output=16, dilation=4, prefix='stage3_conv7')
        net.stage3_relu7 = caffe.layers.ReLU(net.stage3_conv7, in_place = True)
        net.stage3_conv8 = residual_block(net.stage3_relu7, num_output=16, dilation=8, prefix='stage3_conv8')
        net.stage3_relu8 = caffe.layers.ReLU(net.stage3_conv8, in_place = True)
        net.stage3_conv9 = residual_block(net.stage3_relu8, num_output=16, prefix='stage3_conv9')
        net.stage3_relu9 = caffe.layers.ReLU(net.stage3_conv9, in_place = True)
        net.stage3_pool = caffe.layers.Pooling(net.stage3_relu9, pool=caffe.params.Pooling.MAX,  kernel_size=2, stride=2, pad=0)



        # Stage 4
        net.stage4_conv1 = residual_block(net.stage3_relu9, num_output=16, prefix='stage4_conv1')
        net.stage4_relu1 = caffe.layers.ReLU(net.stage4_conv1, in_place = True)
        net.stage4_conv2 = residual_block(net.stage4_relu1, num_output=16, prefix='stage4_conv2')
        net.stage4_relu2 = caffe.layers.ReLU(net.stage4_conv2, in_place = True)
        net.stage4_conv3 = residual_block(net.stage4_relu2, num_output=16, prefix='stage4_conv3')
        net.stage4_relu3 = caffe.layers.ReLU(net.stage4_conv3, in_place = True)
        net.stage4_conv4 = residual_block(net.stage4_relu3, num_output=16, dilation=2, prefix='stage4_conv4')
        net.stage4_relu4 = caffe.layers.ReLU(net.stage4_conv4, in_place = True)
        net.stage4_conv5 = residual_block(net.stage4_relu4, num_output=16, dilation=2, prefix='stage4_conv5')
        net.stage4_relu5 = caffe.layers.ReLU(net.stage4_conv5, in_place = True)
        net.stage4_conv6 = residual_block(net.stage4_relu5, num_output=16, dilation=4, prefix='stage4_conv6')
        net.stage4_relu6 = caffe.layers.ReLU(net.stage4_conv6, in_place = True)
        net.stage4_conv7 = residual_block(net.stage4_relu6, num_output=16, dilation=4, prefix='stage4_conv7')
        net.stage4_relu7 = caffe.layers.ReLU(net.stage4_conv7, in_place = True)
        net.stage4_conv8 = residual_block(net.stage4_relu7, num_output=16, dilation=8, prefix='stage4_conv8')
        net.stage4_relu8 = caffe.layers.ReLU(net.stage4_conv8, in_place = True)
        net.stage4_conv9 = residual_block(net.stage4_relu8, num_output=16, dilation=8, prefix='stage4_conv9')
        net.stage4_relu9 = caffe.layers.ReLU(net.stage4_conv9, in_place = True)
        net.stage4_conv10 = residual_block(net.stage4_relu9, num_output=16, prefix='stage4_conv10')
        net.stage4_relu10 = caffe.layers.ReLU(net.stage4_conv10, in_place = True)
        net.stage4_pool = caffe.layers.Pooling(net.stage4_relu10, pool=caffe.params.Pooling.MAX, kernel_size=2, stride=2, pad=0)

        # Decoding 1
        net.decoding1_s2d_conv1 = caffe.layers.Convolution(net.stage1_relu4, kernel_size = 3, pad=1, num_output = 8,
                                             weight_filler = dict(type = 'constant', value=0))
        net.decoding1_s2d_relu1 = caffe.layers.ReLU(net.decoding1_s2d_conv1, in_place = True)
        net.decoding1_s2d_conv2 = caffe.layers.Convolution(net.decoding1_s2d_relu1, kernel_size = 3, pad=1, num_output = 4,
                                             weight_filler = dict(type = 'constant', value=0))
        net.decoding1_s2d_relu2 = caffe.layers.ReLU(net.decoding1_s2d_conv2, in_place = True)
        net.decoding1_s2d_conv3 = caffe.layers.Convolution(net.decoding1_s2d_relu2, kernel_size = 3, pad=1, num_output = 1,
                                             weight_filler = dict(type = 'constant', value=0))

        net.decoding1_d2s_conv1 = caffe.layers.Convolution(net.stage1_relu4, kernel_size = 3, pad=1, num_output = 8,
                                             weight_filler = dict(type = 'constant', value=0))
        net.decoding1_d2s_relu1 = caffe.layers.ReLU(net.decoding1_d2s_conv1, in_place = True)
        net.decoding1_d2s_conv2 = caffe.layers.Convolution(net.decoding1_d2s_relu1, kernel_size = 3, pad=1, num_output = 4,
                                             weight_filler = dict(type = 'constant', value=0))
        net.decoding1_d2s_relu2 = caffe.layers.ReLU(net.decoding1_d2s_conv2, in_place = True)
        net.decoding1_d2s_conv3 = caffe.layers.Convolution(net.decoding1_d2s_relu2, kernel_size = 3, pad=1, num_output = 1,
                                             weight_filler = dict(type = 'constant', value=0))


        # Decoding 2
        net.decoding2_s2d_conv1 = caffe.layers.Convolution(net.stage2_relu6, kernel_size = 3, pad=1, num_output = 8,
                                             weight_filler = dict(type = 'constant', value=0))
        net.decoding2_s2d_relu1 = caffe.layers.ReLU(net.decoding2_s2d_conv1, in_place = True)
        net.decoding2_s2d_conv2 = caffe.layers.Convolution(net.decoding2_s2d_relu1, kernel_size = 3, pad=1, num_output = 4,
                                             weight_filler = dict(type = 'constant', value=0))
        net.decoding2_s2d_relu2 = caffe.layers.ReLU(net.decoding2_s2d_conv2, in_place = True)
        net.decoding2_s2d_conv3 = caffe.layers.Convolution(net.decoding2_s2d_relu2, kernel_size = 3, pad=1, num_output = 1,
                                             weight_filler = dict(type = 'constant', value=0))

        net.decoding2_d2s_conv1 = caffe.layers.Convolution(net.stage2_relu6, kernel_size = 3, pad=1, num_output = 8,
                                             weight_filler = dict(type = 'constant', value=0))
        net.decoding2_d2s_relu1 = caffe.layers.ReLU(net.decoding2_d2s_conv1, in_place = True)
        net.decoding2_d2s_conv2 = caffe.layers.Convolution(net.decoding2_d2s_relu1, kernel_size = 3, pad=1, num_output = 4,
                                             weight_filler = dict(type = 'constant', value=0))
        net.decoding2_d2s_relu2 = caffe.layers.ReLU(net.decoding2_d2s_conv2, in_place = True)
        net.decoding2_d2s_conv3 = caffe.layers.Convolution(net.decoding2_d2s_relu2, kernel_size = 3, pad=1, num_output = 1,
                                             weight_filler = dict(type = 'constant', value=0))



        # Decoding 3
        net.decoding3_s2d_conv1 = caffe.layers.Convolution(net.stage3_relu9, kernel_size = 3, pad=1, num_output = 8,
                                             weight_filler = dict(type = 'constant', value=0))
        net.decoding3_s2d_relu1 = caffe.layers.ReLU(net.decoding3_s2d_conv1, in_place = True)
        net.decoding3_s2d_conv2 = caffe.layers.Convolution(net.decoding3_s2d_relu1, kernel_size = 3, pad=1, num_output = 4,
                                             weight_filler = dict(type = 'constant', value=0))
        net.decoding3_s2d_relu2 = caffe.layers.ReLU(net.decoding3_s2d_conv2, in_place = True)
        net.decoding3_s2d_conv3 = caffe.layers.Convolution(net.decoding3_s2d_relu2, kernel_size = 3, pad=1, num_output = 1,
                                             weight_filler = dict(type = 'constant', value=0))

        net.decoding3_d2s_conv1 = caffe.layers.Convolution(net.stage3_relu9, kernel_size = 3, pad=1, num_output = 8,
                                             weight_filler = dict(type = 'constant', value=0))
        net.decoding3_d2s_relu1 = caffe.layers.ReLU(net.decoding3_d2s_conv1, in_place = True)
        net.decoding3_d2s_conv2 = caffe.layers.Convolution(net.decoding3_d2s_relu1, kernel_size = 3, pad=1, num_output = 4,
                                             weight_filler = dict(type = 'constant', value=0))
        net.decoding3_d2s_relu2 = caffe.layers.ReLU(net.decoding3_d2s_conv2, in_place = True)
        net.decoding3_d2s_conv3 = caffe.layers.Convolution(net.decoding3_d2s_relu2, kernel_size = 3, pad=1, num_output = 1,
                                             weight_filler = dict(type = 'constant', value=0))



        # Decoding 4
        net.decoding4_s2d_conv1 = caffe.layers.Convolution(net.stage4_relu10, kernel_size = 3, pad=1, num_output = 8,
                                             weight_filler = dict(type = 'constant', value=0))
        net.decoding4_s2d_relu1 = caffe.layers.ReLU(net.decoding4_s2d_conv1, in_place = True)
        net.decoding4_s2d_conv2 = caffe.layers.Convolution(net.decoding4_s2d_relu1, kernel_size = 3, pad=1, num_output = 4,
                                             weight_filler = dict(type = 'constant', value=0))
        net.decoding4_s2d_relu2 = caffe.layers.ReLU(net.decoding4_s2d_conv2, in_place = True)
        net.decoding4_s2d_conv3 = caffe.layers.Convolution(net.decoding4_s2d_relu2, kernel_size = 3, pad=1, num_output = 1,
                                             weight_filler = dict(type = 'constant', value=0))

        net.decoding4_d2s_conv1 = caffe.layers.Convolution(net.stage4_relu10, kernel_size = 3, pad=1, num_output = 8,
                                             weight_filler = dict(type = 'constant', value=0))
        net.decoding4_d2s_relu1 = caffe.layers.ReLU(net.decoding4_d2s_conv1, in_place = True)
        net.decoding4_d2s_conv2 = caffe.layers.Convolution(net.decoding4_d2s_relu1, kernel_size = 3, pad=1, num_output = 4,
                                             weight_filler = dict(type = 'constant', value=0))
        net.decoding4_d2s_relu2 = caffe.layers.ReLU(net.decoding4_d2s_conv2, in_place = True)
        net.decoding4_d2s_conv3 = caffe.layers.Convolution(net.decoding4_d2s_relu2, kernel_size = 3, pad=1, num_output = 1,
                                             weight_filler = dict(type = 'constant', value=0))

        net.deconv2_s2d = caffe.layers.Deconvolution(net.decoding2_s2d_conv3, convolution_param=dict(num_output=1, kernel_size=2, stride=2), param=[dict(lr_mult=1, decay_mult=1)])
        net.deconv2_d2s = caffe.layers.Deconvolution(net.decoding2_d2s_conv3, convolution_param=dict(num_output=1, kernel_size=2, stride=2), param=[dict(lr_mult=1, decay_mult=1)])

        net.deconv3_s2d = caffe.layers.Deconvolution(net.decoding3_s2d_conv3, convolution_param=dict(num_output=1, kernel_size=2, stride=2), param=[dict(lr_mult=1, decay_mult=1)])
        net.deconv3_d2s = caffe.layers.Deconvolution(net.decoding3_d2s_conv3, convolution_param=dict(num_output=1, kernel_size=2, stride=2), param=[dict(lr_mult=1, decay_mult=1)])

        net.deconv4_s2d = caffe.layers.Deconvolution(net.decoding4_s2d_conv3, convolution_param=dict(num_output=1, kernel_size=2, stride=2), param=[dict(lr_mult=1, decay_mult=1)])
        net.deconv4_d2s = caffe.layers.Deconvolution(net.decoding4_d2s_conv3, convolution_param=dict(num_output=1, kernel_size=2, stride=2), param=[dict(lr_mult=1, decay_mult=1)])        # Fusion
        net.fusion1 = caffe.layers.Concat(net.decoding1_s2d_conv3, net.decoding1_d2s_conv3, axis=1)
        net.fusion2 = caffe.layers.Concat(net.decoding2_s2d_conv3, net.decoding2_d2s_conv3, axis=1)
        net.fusion3 = caffe.layers.Concat(net.decoding3_s2d_conv3, net.decoding3_d2s_conv3, axis=1)
        net.fusion4 = caffe.layers.Concat(net.decoding4_s2d_conv3, net.decoding4_d2s_conv3, axis=1)
        net.fusion5 = caffe.layers.Concat(net.fusion1, net.fusion2, axis=1)
        net.fusion6 = caffe.layers.Concat(net.fusion5, net.fusion3, axis=1)
        net.fusion7 = caffe.layers.Concat(net.fusion6, net.fusion4, axis=1)




        net.fusion = caffe.layers.Convolution(net.fusion7, kernel_size = 1, num_output = 1,
                                             weight_filler = dict(type = 'xavier'))
        net.concat = caffe.layers.Concat(net.fusion, net.fusion, axis=1)
        net.concat2 = caffe.layers.Concat(net.fusion, net.concat, axis=1)

        net.loss =  caffe.layers.SigmoidCrossEntropyLoss(net.concat2, net.labels)
        
        return net.to_proto()
    
    def count_errors(scores, labels):
        """
        Utility method to count the errors given the ouput of the
        "score" layer and the labels.
        
        :param score: output of score layer
        :type score: numpy.ndarray
        :param labels: labels
        :type labels: numpy.ndarray
        :return: count of errors
        :rtype: int
        """
        return numpy.mean(numpy.abs(scores - labels)) 
    
    assert os.path.exists(args.train_lmdb), "LMDB %s does not exist" % args.train_lmdb
    assert os.path.exists(args.test_lmdb), "LMDB %s does not exist" % args.test_lmdb
    assert os.path.exists(args.train_gt_lmdb), "LMDB %s does not exist" % args.train_gt_lmdb
    assert os.path.exists(args.test_gt_lmdb), "LMDB %s does not exist" % args.test_gt_lmdb

    env_train = lmdb.open(args.train_lmdb)
    txn_train = env_train.begin()
    env_test = lmdb.open(args.test_lmdb)
    txn_test = env_test.begin()

    train_sizes = txn_train.stat()['entries']
    test_sizes = txn_test.stat()['entries']
    
    train_prototxt_path = args.working_directory + '/train.prototxt'
    test_prototxt_path = args.working_directory + '/test.prototxt'
    deploy_prototxt_path = args.working_directory + '/deploy.prototxt'

    print("Start building network")

    #with open(train_prototxt_path, 'w') as f:
    #    f.write(str(network(args.train_lmdb, args.train_gt_lmdb, 1)))
    
    #with open(test_prototxt_path, 'w') as f:
    #    f.write(str(network(args.test_lmdb, args.test_gt_lmdb, 1)))
    
    tools.prototxt.train2deploy(train_prototxt_path, (1, 3, 481, 321), deploy_prototxt_path)


    prototxt_solver = args.working_directory + '/solver.prototxt'
    solver_prototxt = tools.solvers.SolverProtoTXT({
        'train_net': train_prototxt_path,
        'test_net': test_prototxt_path,
        'test_initialization': 'false', # no testing
        'test_iter': 0, # no testing
        'test_interval': 100000,
        'base_lr': 0.001,
        'lr_policy': 'step',
        'gamma': 0.01,
        'stepsize': 1000,
        'display': 100,
        'max_iter': 1000,
        'momentum': 0.95,
        'weight_decay': 0.005,
        'snapshot': 0, # only at the end
        'snapshot_prefix': args.working_directory + '/snapshot',
        'solver_mode': 'CPU'
    })
    
    solver_prototxt.write(prototxt_solver)
    solver = caffe.SGDSolver(prototxt_solver)

    print("Finished building optimizer")

    callbacks = []
    
    # Callback to report loss in console. Also automatically plots the loss
    # and writes it to the given file. In order to silence the console,
    # use plot_loss instead of report_loss.
    report_loss = tools.solvers.PlotLossCallback(100, args.working_directory + '/loss.png')
    callbacks.append({
        'callback': tools.solvers.PlotLossCallback.report_loss,
        'object': report_loss,
        'interval': 1,
    })
    
    # Callback to report error in console.
    report_error = tools.solvers.PlotErrorCallback(count_errors, train_sizes, test_sizes, 
                                                   solver_prototxt.get_parameters()['snapshot_prefix'], 
                                                   args.working_directory + '/error.png')
    callbacks.append({
        'callback': tools.solvers.PlotErrorCallback.report_error,
        'object': report_error,
        'interval': 500,
    })
    
    # Callback to save an "early stopping" model.
    callbacks.append({
        'callback': tools.solvers.PlotErrorCallback.stop_early,
        'object': report_error,
        'interval': 500,
    })
    
    # Callback for reporting the gradients for all layers in the console.
    report_gradient = tools.solvers.PlotGradientCallback(100, args.working_directory + '/gradient.png')
    callbacks.append({
        'callback': tools.solvers.PlotGradientCallback.report_gradient,
        'object': report_gradient,
        'interval': 1,
    })   
    
    # Callback for saving regular snapshots using the snapshot_prefix in the
    # solver prototxt file.
    # Is added after the "early stopping" callback to avoid problems.
    callbacks.append({
        'callback': tools.solvers.SnapshotCallback.write_snapshot,
        'object': tools.solvers.SnapshotCallback(),
        'interval': 500,
    })
    
    monitoring_solver = tools.solvers.MonitoringSolver(solver)
    monitoring_solver.register_callback(callbacks)
    monitoring_solver.solve(args.iterations)

def main_resume():
    """
    Resume training; assumes training has been started using :func:`examples.bsds500.main_train`.
    """
    
    def count_errors(scores, labels):
        """
        Utility method to count the errors given the ouput of the
        "score" layer and the labels.
        
        :param score: output of score layer
        :type score: numpy.ndarray
        :param labels: labels
        :type labels: numpy.ndarray
        :return: count of errors
        :rtype: int
        """
        
        return numpy.sum(numpy.argmax(scores, axis = 1) != labels)   
        
    max_iteration = 0
    files = glob.glob(args.working_directory + '/*.solverstate')
    
    for filename in files:
        filenames = filename.split('_')
        iteration = filenames[-1][:-12]
        
        try:
            iteration = int(iteration)
            if iteration > max_iteration:
                max_iteration = iteration
        except:
            pass
    
    caffemodel = args.working_directory + '/snapshot_iter_' + str(max_iteration) + '.caffemodel'
    solverstate = args.working_directory + '/snapshot_iter_' + str(max_iteration) + '.solverstate'
    
    train_prototxt_path = args.working_directory + '/train.prototxt'
    test_prototxt_path = args.working_directory + '/test.prototxt'
    deploy_prototxt_path = args.working_directory + '/deploy.prototxt'
    solver_prototxt_path = args.working_directory + '/solver.prototxt'
    
    assert max_iteration > 0, "could not find a solverstate or snaphot file to resume"
    assert os.path.exists(caffemodel), "caffemodel %s not found" % caffemodel
    assert os.path.exists(solverstate), "solverstate %s not found" % solverstate
    assert os.path.exists(train_prototxt_path), "prototxt %s not found" % train_prototxt_path
    assert os.path.exists(test_prototxt_path), "prototxt %s not found" % test_prototxt_path
    assert os.path.exists(deploy_prototxt_path), "prototxt %s not found" % deploy_prototxt_path
    assert os.path.exists(solver_prototxt_path), "prototxt %s not found" % solver_prototxt_path
    
    solver = caffe.SGDSolver(solver_prototxt_path)
    solver.restore(solverstate)
    
    solver.net.copy_from(caffemodel)
    
    solver_prototxt = tools.solvers.SolverProtoTXT()
    solver_prototxt.read(solver_prototxt_path)     
    callbacks = []
    
    # Callback to report loss in console.
    report_loss = tools.solvers.PlotLossCallback(100, args.working_directory + '/loss.png')
    callbacks.append({
        'callback': tools.solvers.PlotLossCallback.report_loss,
        'object': report_loss,
        'interval': 1,
    })
    
    # Callback to report error in console.
    report_error = tools.solvers.PlotErrorCallback(count_errors, 60000, 10000, 
                                                   solver_prototxt.get_parameters()['snapshot_prefix'], 
                                                   args.working_directory + '/error.png')
    callbacks.append({
        'callback': tools.solvers.PlotErrorCallback.report_error,
        'object': report_error,
        'interval': 500,
    })
    
    # Callback to save an "early stopping" model.
    callbacks.append({
        'callback': tools.solvers.PlotErrorCallback.stop_early,
        'object': report_error,
        'interval': 500,
    })
    
    # Callback for reporting the gradients for all layers in the console.
    report_gradient = tools.solvers.PlotGradientCallback(100, args.working_directory + '/gradient.png')
    callbacks.append({
        'callback': tools.solvers.PlotGradientCallback.report_gradient,
        'object': report_gradient,
        'interval': 1,
    })
    
    # Callback for saving regular snapshots using the snapshot_prefix in the
    # solver prototxt file.
    # Is added after the "early stopping" callback to avoid problems.
    callbacks.append({
        'callback': tools.solvers.SnapshotCallback.write_snapshot,
        'object': tools.solvers.SnapshotCallback(),
        'interval': 500,
    })  
    
    monitoring_solver = tools.solvers.MonitoringSolver(solver, max_iteration)
    monitoring_solver.register_callback(callbacks)
    monitoring_solver.solve(args.iterations)
    
def main_detect():
    """
    Detect edges on a given image, after training a network using :func:`examples.bsds500.main_train`.
    """
    
    pass

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    if args.mode == 'extract':
        main_extract()
    if args.mode == 'subsample_test':
        main_subsample_test()
    elif args.mode == 'train':
        main_train()
    elif args.mode =='resume':
        main_resume()
    else:
        print('Invalid mode.')
