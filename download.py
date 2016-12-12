#!usr/local/python3
'''
Download the notMNIST dataset from Internet
Skip it if the dataset is already in the direction
'''

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from urllib.request import urlretrieve
import pickle

url = 'http://commondatastorage.googleapis.com/books1000/'
last_percentage_reported = None

def reporthook(count, blockSize, totalSize):
    '''A hook to report the progress of a download'''
    readSoFar = count * blockSize
    if totalSize > 0:
        percent = readSoFar * 100 / totalSize
        s = '\r{:>5.0f}% {} / {}'.format(percent, readSoFar, totalSize)
        sys.stdout.write(s)
        sys.stdout.flush()
        if readSoFar >= totalSize: # the end
            sys.stdout.write('\n')
    else: # total size is unknown
        sys.stdout.write('read {}\n'.format(readSoFar))

def maybe_download(filename, expected_bytes, force = False):
    '''Download a file if is not in the directory and
    make sure it is the right size'''
    if force or not os.path.exists(filename):
        # override the exsited files by setting force = True
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url + filename, filename,
            reporthook)
        print('Download Complete')
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print('Failed to verify {}.'.format(filename))
        print('Attempt to redownload it.')
        filename, _ = urlretrieve(url + filename, filename,
            reporthook)
        print('Download Complete')
    return filename

def main():
    train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
    test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

if __name__ == '__main__': main()






