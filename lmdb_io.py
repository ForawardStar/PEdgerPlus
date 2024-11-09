"""
Examples for reading LMDBs.

.. argparse::
   :ref: examples.lmdb_io.get_parser
   :prog: lmdb_io
"""

import os
import cv2
import argparse

# To silence Caffe! Must be added before importing Caffe or modules which
# are importing Caffe.
os.environ['GLOG_minloglevel'] = '0'
import tools.lmdb_io

def get_parser():
    """
    Get the parser.
    
    :return: parser
    :rtype: argparse.ArgumentParser
    """
    
    parser = argparse.ArgumentParser(description = 'Read LMDBs.')
    parser.add_argument('--mode', default = 'write')
    parser.add_argument('--lmdb', default = 'examples/bsds500/train_lmdb', type = str,
                       help = 'path to input LMDB')
    parser.add_argument('--output', default = 'examples/output', type = str,
                        help = 'output directory')
    parser.add_argument('--limit', default = 100, type = int,
                        help = 'limit the number of images to read')
                       
    return parser

def main_statistics():
    """
    Read and print the size of an LMDB.
    """
    
    lmdb = tools.lmdb_io.LMDB(args.lmdb)
    print(lmdb.count())

def main_read():
    """
    Read up to ``--limit`` images from the LMDB.
    """
    
    lmdb = tools.lmdb_io.LMDB(args.lmdb)
    keys = lmdb.keys()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    with open(args.output + '/labels.txt', 'w') as f:
        for n in range(min(len(keys), args.limit)):
            image, label, key = lmdb.read_single(keys[n])
            image_path = args.output + '/' + keys[n] + '.png'
            cv2.imwrite(image_path, image)
            f.write(image_path + ': ' + str(label) + '\n')

def main_write():
    """
    Read up to ``--limit`` images from the LMDB.
    """

    lmdb = tools.lmdb_io.LMDB(args.lmdb)

    scale_type = ['aug_data', 'aug_data_scale_0.5', 'aug_data_scale_1.5']

    rotate_type = ['0.0_1_0',  '112.5_1_0',  '135.0_1_0',  '157.5_1_0',  '180.0_1_0',  '202.5_1_0',  '225.0_1_0',  '22.5_1_0',  '247.5_1_0',  '270.0_1_0',  '292.5_1_0',  '315.0_1_0',  '337.5_1_0',  '45.0_1_0',  '67.5_1_0',  '90.0_1_0', '0.0_1_1',  '112.5_1_1',  '135.0_1_1',  '157.5_1_1',  '180.0_1_1',  '202.5_1_1',  '225.0_1_1',  '22.5_1_1',  '247.5_1_1',  '270.0_1_1',  '292.5_1_1',  '315.0_1_1',  '337.5_1_1',  '45.0_1_1',  '67.5_1_1',  '90.0_1_1']

    filenames = os.listdir("/home/fyb/HED-BSDS/train/aug_data/0.0_1_0/")

    data_root = "/home/fyb/HED-BSDS/train/"

    images = []
    edges = []
    for scale in scale_type:
        for rotate in rotate_type:
            for filename in filenames:
                data_path = data_root + scale + "/" + rotate + "/"+ filename
                gt_path = data_root + scale.replace("data", "gt") + "/" + rotate + "/"+ filename.replace(".jpg", ".png")
                images.append(cv2.imread(data_path))
                edges.append(cv2.imread(gt_path))
                
    print("Finishing reading images")

    lmdb.write(images, edges)


def main_write_test():
    """
    Read up to ``--limit`` images from the LMDB.
    """

    lmdb = tools.lmdb_io.LMDB(args.lmdb)

    image_root = "/home/fyb/HED-BSDS/test/"
    gt_root = "/home/fyb/HED-BSDS/groundTruth_png/"

    filenames = os.listdir(image_root)

    images = []
    edges = []
    for filename in filenames:
        data_path = image_root + filename
        gt_path =  gt_root + filename.replace(".jpg", ".png")
        images.append(cv2.imread(data_path))
        edges.append(cv2.imread(gt_path, 0))
    print("Finishing reading images")

    lmdb.write(images, edges, 'TEST')

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    if args.mode == 'read':
        main_read()
    elif args.mode == 'write':
        main_write()
    elif args.mode == 'write_test':
        main_write_test()
    elif args.mode == 'statistics':
        main_statistics()
    else:
        print('Invalid mode.')
