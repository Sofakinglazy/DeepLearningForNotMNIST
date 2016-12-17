#!usr/local/python3
'''
    Apply logisticRegression model onto the notMNIST data
'''

from sklearn.linear_model import LogisticRegression
import numpy as np
import os
import pickle
from config_parser import write_config, read_config

config_file = 'config.ini'
# if not config_file:
write_config()

image_size, train_size, valid_size, test_size = read_config()
pickle_file = 'notMNIST.pickle'
'''
names = {
        'train_dataset' : train_dataset,
        'valid_dataset' : valid_dataset,
        'test_dataset'  : test_dataset,
        'train_labels'  : train_labels,
        'valid_labels'  : valid_labels,
        'test_labels'   : test_labels
    }
'''
'''
    read pickle file and attain training, validation
    and testing sets
'''
def read_pickle():
    if not os.path.exists(pickle_file):
        raise Exception('file {} not exists'.format(pickle_file))
    with open(pickle_file, 'rb') as f:
        pack = pickle.load(f)
        train_dataset = pack.get('train_dataset')
        valid_dataset = pack.get('valid_dataset')
        test_dataset  = pack.get('test_dataset')
        train_labels  = pack.get('train_labels')
        valid_labels  = pack.get('valid_labels')
        test_labels   = pack.get('test_labels')
    return train_dataset, valid_dataset, test_dataset, \
    train_labels, valid_labels, test_labels

def flatten_array(array):
    if array.ndim <= 2:
        return array
    num_features = 1
    num_samples = len(array[:, 0])
    for i in range(1, array.ndim):
        num_features *= len(array[i, :]) # num of features
    new_array = np.ndarray((num_samples, num_features), dtype = np.float32)
    for i in range(0, num_samples):
        new_array[i, :] = array[i, :, :].ravel()
    print('Done transform')
    print(new_array.shape)
    return new_array

def main():
    train_dataset, valid_dataset, test_dataset, \
        train_labels, valid_labels, test_labels = read_pickle()
    print(train_dataset.shape)
    # train_dataset = np.ndarray.reshape(train_dataset, (train_size, 1, image_size*image_size))
    # print(train_dataset.shape)
    model = LogisticRegression()
    print('Model has been created')
    _train_dataset = flatten_array(train_dataset)
    _train_labels = flatten_array(train_labels)
    _test_dataset = flatten_array(test_dataset)
    _test_labels = flatten_array(test_labels)
    model = model.fit(_train_dataset, _train_labels)
    print('Done fitting the model')
    score = model.score(_test_dataset, _test_labels)
    print(score)

if __name__ == '__main__': main()




