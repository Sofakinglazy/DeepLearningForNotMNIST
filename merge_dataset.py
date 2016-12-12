#!usr/local/python3
'''
    format the training set and testing set
'''

import numpy as np
import scipy
import pickle
from download import maybe_download
from extract import maybe_extract
from read_image import maybe_pickle

image_size = 28         # image width and height
train_size = 200000     # training set size
valid_size = 10000      # validation set size
test_size = 10000       # testing set size

def make_array(rows, img_size):
    if rows:
        dataset = np.ndarray((rows, img_size, img_size), dtype = np.float32)
        labels = np.ndarray(rows, dtype = np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size = 0):
    num_classes = len(pickle_files)
    train_dataset, train_labels = make_array(train_size, image_size)
    valid_dataset, valid_labels = make_array(valid_size, image_size)
    tsize_per_class = train_size // num_classes
    vsize_per_class = valid_size // num_classes

    start_t, start_v = 0, 0
    end_t, end_v = tsize_per_class, vsize_per_class
    end_l = tsize_per_class + vsize_per_class # total length
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                img_set = pickle.load(f)
                # shuffle the imagesto have random training and validation set
                np.random.shuffle(img_set)
                # for validation set
                if valid_dataset is not None:
                    valid_image = img_set[:vsize_per_class, :, :]   # first set as validation
                    print(start_v, end_v)
                    valid_dataset[start_v:end_v, :, :] = valid_image
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_image = img_set[vsize_per_class:end_l, :, :]  # the rest for training set
                train_dataset[start_t:end_t, :, :] = train_image
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process the data form {}: e'.format(
                    pickle_file, e))
            raise
    return train_dataset, train_labels, valid_dataset, valid_labels

def main():
    train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
    test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)
    train_folders = maybe_extract(train_filename)
    test_folders = maybe_extract(test_filename)
    train_datasets = maybe_pickle(train_folders, 45000)
    test_datasets = maybe_pickle(test_folders, 1800)
    train_dataset, train_labels, valid_dataset, valid_labels = merge_datasets(
        train_datasets, train_size, valid_size)
    test_dataset, test_labels, _, _ = merge_datasets(
        test_datasets, test_size)
    print('Training:', train_dataset.shape, train_labels.shape)
    print('Validation:', valid_dataset.shape, valid_labels.shape)
    print('Test:', test_dataset.shape, test_labels.shape)

if __name__ == '__main__': main()