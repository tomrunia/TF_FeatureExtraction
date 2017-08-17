# MIT License
# 
# Copyright (c) 2017 Tom Runia
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2017-08-15

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from datasets import imagenet

# ImageNet mapping class_index => class_name
imagenet_classnames = imagenet.create_readable_names_for_imagenet_labels()


def find_files(paths, extensions, sort=True):
    '''
    Returns a list of files in one or multiple directories.
    :param paths: str or list, paths to search in for files
    :param extensions: str or list, extensions to match
    :param sort: bool, whether to sort the list of found files
    :return: list of (sorted) files that are found
    '''
    if type(paths) is str:
        paths = [paths]
    files = []
    for path in paths:
        for file in os.listdir(path):
            if file.endswith(extensions):
                files.append(os.path.join(path, file))
    if sort:
        files.sort()
    return files

def fill_last_batch(image_list, batch_size):
    '''
    Fill up the last batch with the last example for the list.
    Operation is performed in-place.

    :param image_list: list of str, image list to fill up
    :param batch_size: int, batch_size
    :return:
    '''
    num_examples = len(image_list)
    num_batches = int(np.ceil(num_examples/batch_size))
    for i in range((num_batches*batch_size)-num_examples):
        image_list.append(image_list[-1])

def sort_feature_dataset(feature_dataset):
    '''
    When more than one preprocessing thread is used the feature_dataset is
    not sorted according to alphabetical order of filenames. This function
    sorts the dataset in place so that filenames and corresponding fetaures
    are sorted by its filename. Note: sorting is in-place.

    :param feature_dataset: dict, containting filenames and all features
    :return:
    '''
    indices = np.argsort(feature_dataset['filenames'])
    feature_dataset['filenames'].sort()
    # Apply sorting to features for each image
    for key in feature_dataset.keys():
        if key == 'filenames': continue
        feature_dataset[key] = feature_dataset[key][indices]

def write_hdf5(filename, layer_names, feature_dataset):
    '''
    Writes features to HDF5 file.
    :param filename: str, filename to output
    :param layer_names: list of str, layer names
    :param feature_dataset: dict, containing features[layer_names] = vals
    :return:
    '''
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset("filenames", data=feature_dataset['filenames'])
        for layer_name in layer_names:
            hf.create_dataset(layer_name, data=feature_dataset[layer_name], dtype=np.float32)

def display_imagenet_prediction(image, class_index):
    class_label = imagenet_classnames[class_index]
    print("Prediction: {} (class_index={})".format(class_label, class_index))
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.show()


