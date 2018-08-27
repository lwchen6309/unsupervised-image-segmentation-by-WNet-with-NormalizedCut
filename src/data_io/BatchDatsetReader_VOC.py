"""
Revised from [FCN code by shekkizh] (https://github.com/shekkizh/FCN.tensorflow)
"""

import os, sys
import pickle
from glob import glob
from tensorflow.python.platform import gfile

import numpy as np
import imageio
import scipy.misc as misc
from skimage import color
# import matplotlib.pyplot as plt

from six.moves import urllib
import tarfile, zipfile


def create_image_lists(image_dir):
    """
    Read 'image_dir/*/training' and 'image_dir/*/validation'
    into list of dict with keys: 'image', 'annotation', 'filename'
    """
    
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    
    # Find image list if its common in annotation
    image_pattern = os.path.join(image_dir, 'JPEGImages', '*.' + 'jpg')
    image_lst = glob(image_pattern)        
    data = []
    if not image_lst:
        print('No files found')    
    else:
        for image_file in image_lst:
            filename = image_file.split("/")[-1].split('.')[0]
            annotation_file = os.path.join(image_dir, 'SegmentationClass', filename + '.png')
            if os.path.exists(annotation_file):
                record = {'image': image_file, 'annotation': annotation_file, 'filename': filename}
                data.append(record)
            else:
                print('Annotation file not found for %s - Skipping' % filename)
                print('Pattern %s' % annotation_file)
                
    print ('Nunmber of files: %d' %len(data))
    return data
        
def read_data_record(data_dir, validation_len = 500):
    """
    Initialize list of datapath in data_dir if has not been initialized.
    """
    
    pickle_filename = 'VOC_datalist.pickle'
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        data = create_image_lists(data_dir)
        # Parse data into training and validation
        training_data = data[validation_len:]
        validation_data = data[:validation_len]
        result = {'training':training_data, 'validation':validation_data}
        
        print ('Pickling ...')
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ('Found pickle file!')

    with open(pickle_filepath, 'rb') as f:
        data_records = pickle.load(f)
    return data_records

def download_if_no_data(dir_path, url_name):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = url_name.split('/')[-1]
    filepath = os.path.join(dir_path, filename)
    
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        print('Start download to %s'%filepath)
        filepath, _ = urllib.request.urlretrieve(url_name, filepath, reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    else:
        print('Found url file %s.'%filepath)
    
    if tarfile.is_tarfile(filepath):
        tarfile.open(filepath, 'r').extractall(dir_path)
    elif zipfile.is_zipfile(filepath):
        with zipfile.ZipFile(filepath) as zf:
            zip_dir = zf.namelist()[0]
            zf.extractall(dir_path)
    return

def create_BatchDatset():
    print("Download if not VOC2012 exist...")
    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    download_if_no_data('./data/', url)
    
    print("Initializing VOC2012 Batch Dataset Reader...")
    data_record = read_data_record('./data/VOCdevkit/VOC2012')
    train_dataset = BatchDatset(data_record['training'], True)
    valid_dataset = BatchDatset(data_record['validation'], False)
    
    return train_dataset, valid_dataset
    
class BatchDatset:
    
    images = []
    annotations = []
    batch_offset = 0
    epochs_completed = 0
    
    def __init__(self, data_records, is_shuffle=False):
        """
        
        """
        
        print("Initializing Batch Dataset Reader...")
        self.read_data_to_self(data_records)
        
        if is_shuffle:
            self.shuffle_data()
        return
    
    def read_data_to_self(self, data_records, resize_size = 128):
        self.images = np.stack([misc.imresize(imageio.imread(datum['image']), 
                        [resize_size, resize_size], interp='bilinear') for datum in data_records])                        
        self.annotations = np.stack([misc.imresize(imageio.imread(datum['annotation']), 
                        [resize_size, resize_size], interp='bilinear') for datum in data_records])
        return
    
    def shuffle_data(self):
        randperm = np.random.permutation(len(self.images))
        self.images = self.images[randperm]
        self.annotations = self.annotations[randperm]
    
    def get_records(self):
        return self.images, self.annotations
    
    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset
    
    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > len(self.images):
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            self.shuffle_data()
            # Start next epoch
            start = 0
            self.batch_offset = batch_size
        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]
    
    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, len(self.images), size=[batch_size])
        return self.images[indexes], self.annotations[indexes]