#!usr/local/python3
'''
    Apply logisticRegression model onto the notMNIST data
'''

from sklearn.linear_model import LogisticRegression
import numpy as np
import os
import pickle

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

def main():
    train_dataset, valid_dataset, test_dataset, \
        train_labels, valid_labels, test_labels = read_pickle()
    print(train_dataset.shape)

if __name__ == '__main__': main()




