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

import utils

import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
from preprocessing import inception_preprocessing
slim = tf.contrib.slim


class FeatureExtractor(object):

    def __init__(self, network_name, checkpoint_path, batch_size, num_classes,
                 image_size=None, preproc_func_name=None, preproc_threads=1):

        self._network_name = network_name
        self._checkpoint_path = checkpoint_path
        self._batch_size = batch_size
        self._num_classes = num_classes
        self._image_size = image_size
        self._preproc_func_name = preproc_func_name
        self._num_preproc_threads = preproc_threads

        self._global_step = tf.train.get_or_create_global_step()

        # Retrieve the function that returns logits and endpoints
        self._network_fn = nets_factory.get_network_fn(
            self._network_name, num_classes=num_classes, is_training=False)

        # Retrieve the model scope from network factory
        self._model_scope = nets_factory.arg_scopes_map[self._network_name]

        # Fetch the default image size
        self._image_size = self._network_fn.default_image_size

        # Setup the input pipeline with a queue of filenames
        self._filename_queue = tf.FIFOQueue(100000, [tf.string], shapes=[[]], name="filename_queue")
        self._pl_image_files = tf.placeholder(tf.string, shape=[None], name="image_file_list")
        self._enqueue_op = self._filename_queue.enqueue_many([self._pl_image_files])
        self._num_in_queue = self._filename_queue.size()

        # Image reader and preprocessing
        self._batch_from_queue = self._preproc_image_batch(
            self._batch_size, num_threads=preproc_threads)

        # Either use the placeholder as inputs or feed from queue
        self._image_batch = tf.placeholder_with_default(
            self._batch_from_queue, shape=[None, self._image_size, self._image_size, 3])

        # Retrieve the logits and network endpoints (for extracting activations)
        # Note: endpoints is a dictionary with endpoints[name] = tf.Tensor
        self._logits, self._endpoints = self._network_fn(self._image_batch)

        # Find the checkpoint file
        checkpoint_path = self._checkpoint_path
        if tf.gfile.IsDirectory(self._checkpoint_path):
          checkpoint_path = tf.train.latest_checkpoint(self._checkpoint_path)

        # Load pre-trained weights into the model
        variables_to_restore = slim.get_variables_to_restore()
        restore_fn = slim.assign_from_checkpoint_fn(
            self._checkpoint_path, variables_to_restore)

        # Start the session and load the pre-trained weights
        self._sess = tf.Session()
        restore_fn(self._sess)

        # Local variables initializer, needed for queues etc.
        self._sess.run(tf.local_variables_initializer())

        # Managing the queues and threads
        self._coord = tf.train.Coordinator()
        self._threads = tf.train.start_queue_runners(coord=self._coord, sess=self._sess)

    def _preproc_image_batch(self, batch_size, num_threads=1):

        if ("resnet_v2" in self._network_name) and (self._preproc_func_name is None):
            raise ValueError("When using ResNet, please perform the pre-processing "
                            "function manually. See here for details: " 
                            "https://github.com/tensorflow/models/tree/master/slim")

        # Read image file from disk and decode JPEG
        reader = tf.WholeFileReader()
        _, image_raw = reader.read(self._filename_queue)
        image = tf.image.decode_jpeg(image_raw, channels=3)
        # Image preprocessing
        preproc_func_name = self._network_name if self._preproc_func_name is None else self._preproc_func_name
        image_preproc_fn = preprocessing_factory.get_preprocessing(preproc_func_name, is_training=False)
        image_preproc = image_preproc_fn(image, self.image_size, self.image_size)
        # Read a batch of preprocessing images from queue
        image_batch = tf.train.batch([image_preproc], batch_size, num_threads=num_threads)
        return image_batch

    def enqueue_image_files(self, image_files):
        '''
        Given a list of input images, feed these to the queue.
        :param image_files: list of str, list of image files to feed to filename queue
        '''
        self._sess.run(self._enqueue_op, feed_dict={self._pl_image_files: image_files})

    def feed_forward_batch(self, layer_names, images=None):

        # List of network operations (activations) to fetch
        fetches = []

        # Check if all layers are available
        available_layers = self.layer_names()
        for layer_name in layer_names:
            if layer_name not in available_layers:
                raise ValueError("Unable to extract features for layer: {}".format(layer_name))
            fetches.append(self._endpoints[layer_name])

        # Manual inputs using placeholder 'images' of shape [N,H,W,C]
        feed_dict = None
        if images is not None:
            feed_dict = {self._image_batch: images}
        else:
            feed_dict = None

        # Actual forward pass through the network
        fetches.append(self._image_batch)
        outputs = self._sess.run(fetches, feed_dict=feed_dict)
        return outputs

    def num_in_queue(self):
        return self._sess.run(self._num_in_queue)

    def layer_names(self):
        return self._endpoints.keys()

    def print_network_summary(self):
        # Print all the activation names and their shape
        for name, tensor in self._endpoints.items():
            print("{} has shape {}".format(name, tensor.shape))

    @property
    def image_size(self):
        return self._image_size

    @property
    def batch_size(self):
        return self._batch_size

    def close(self):
        self._coord.request_stop()
        self._coord.join(self._threads)
        self._sess.close()