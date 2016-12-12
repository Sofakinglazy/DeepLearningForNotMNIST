#!usr/local/python3
'''
Read the images from folders and put them into dataset
'''

import numpy as np
import scipy
import os
import pickle
import shutil
from download import maybe_download
from extract import maybe_extract


image_size = 28 # image width and height
pixel_depth = 255.0 # the max value of each pixel

def load_images(folder, min_num_images):
    '''Load the images for a single letter'''
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape = (len(image_files), image_size, image_size),
                        dtype = np.float32)
    print('Current fold:', folder)
    num_image = 0
    for image in image_files:
        image_file = os.path.join(folder, image) # format each image name
        try:
            image_data = (scipy.ndimage.imread(image_file).astype(float) -
                            pixel_depth / 2) / pixel_depth # (pixel - 128) /128
            image_shape = image_data.shape
            if image_shape != (image_size, image_size):
                raise Exception('Unexpected image size: {}'.format(image_shape))
            dataset[num_image, :, :] = image_data # put data into dataset by indexing
            num_image += 1
        except IOError as e:
            print('Could not read {}: {} and skip it.'.format(image_file, e))
    if num_image < min_num_images:
        raise Exception('Too fewer images than expected. {} < {}'.format(
                        num_image, min_num_images))
    print('Full dataset tensor: ', dataset.shape)
    print('Mean: ', np.mean(dataset))
    print('Standard deviation: ', np.std(dataset))
    return dataset

def tidy_pickles(directory):
    '''
        tidy the pickles into a single folder pickles
    '''
    if
    directory, _ = os.path.split(directory)
    pickle_folder = os.path.join(directory, 'pickles')
    for file in os.listdir(directory):
        if file.endswith('.pickle'):
            # pickles folder not exists
            if not os.path.exists(pickle_folder):
                os.mkdir(pickle_folder)
            # move into pickles file
            file_path = os.path.join(directory, file)
            shutil.move(file_path, pickle_folder)
            print('Move {} to folder'.format(file))

def maybe_pickle(data_folders, min_num_images_per_class, force = False):
    '''
        Transfer the original dataset into required dataset
        with zero mean and 0.5 deviation. And save the object
        as pickle
    '''
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # Overrider the present data by setting force = True
            print('{} is already present. - Skip pickling.'.format(set_filename))
        else:
            print('Pickling {}'.format(set_filename))
            dataset = load_images(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to {}: {}'.format(set_filename, e))

    return dataset_names

def main():
    train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
    test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)
    train_folders = maybe_extract(train_filename)
    test_folders = maybe_extract(test_filename)
    train_datasets = maybe_pickle(train_folders, 45000)
    test_datasets = maybe_pickle(test_folders, 1800)

if __name__ == '__main__': main()

