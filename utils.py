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


def display_imagenet_prediction(image, class_index):
    class_label = imagenet_classnames[class_index]
    print("Prediction: {} (class_index={})".format(class_label, class_index))
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.show()


