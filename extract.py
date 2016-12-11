#!usr/local/python3
'''
The script is to extract the dataset from dowloaded tar file
And delete the tar file after the process
'''

import numpy as np
import os
import tarfile
import sys
from download import maybe_download

num_class = 10
np.random.seed(133)

def maybe_extract(filename, force = False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0] # get rid of tar.gz
    if os.path.isdir(root) and not force:
        print('{} is alread present. Skip extraction of {}'.format(root, filename))
    else:
        print('Extracting data for {}. This may take a while. Please wait.'.format(filename))
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
    data_foladers = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_foladers) != num_class:
        raise Exception('Expected {} folders. Found {} instead.'.format(
            num_class, len(data_foladers)))
    print(data_foladers)
    return data_foladers

def main():
    train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
    test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)
    train_folders = maybe_extract(train_filename)
    test_folders = maybe_extract(test_filename)

if __name__ == '__main__': main()