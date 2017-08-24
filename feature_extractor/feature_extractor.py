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

import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
slim = tf.contrib.slim


class FeatureExtractor(object):

    def __init__(self, network_name, checkpoint_path, batch_size, num_classes,
                 image_size=None, preproc_func_name=None, preproc_threads=2):

        '''
        TensorFlow feature extractor using tf.slim and models/slim.
        Core functionalities are loading network architecture, pretrained weights,
        setting up an image pre-processing function, queues for fast input reading.
        The main workflow after initialization is first loading a list of image
        files using the `enqueue_image_files` function and then pushing them
        through the network with `feed_forward_batch`.

        For pre-trained networks and some more explanation, checkout:
          https://github.com/tensorflow/models/tree/master/slim

        :param network_name: str, network name (e.g. resnet_v1_101)
        :param checkpoint_path: str, full path to checkpoint file to load
        :param batch_size: int, batch size
        :param num_classes: int, number of output classes
        :param image_size: int, width and height to overrule default_image_size (default=None)
        :param preproc_func_name: func, optional to overwrite default processing (default=None)
        :param preproc_threads: int, number of input threads (default=1)

        '''

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
        self._batch_from_queue, self._batch_filenames = \
            self._preproc_image_batch(self._batch_size, num_threads=preproc_threads)

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
        '''
        This function is only used for queue input pipeline. It reads a filename
        from the filename queue, decodes the image, pushes it through a pre-processing
        function and then uses tf.train.batch to generate batches.

        :param batch_size: int, batch size
        :param num_threads: int, number of input threads (default=1)
        :return: tf.Tensor, batch of pre-processed input images
        '''

        if ("resnet_v2" in self._network_name) and (self._preproc_func_name is None):
            raise ValueError("When using ResNet, please perform the pre-processing "
                            "function manually. See here for details: " 
                            "https://github.com/tensorflow/models/tree/master/slim")

        # Read image file from disk and decode JPEG
        reader = tf.WholeFileReader()
        image_filename, image_raw = reader.read(self._filename_queue)
        image = tf.image.decode_jpeg(image_raw, channels=3)
        # Image preprocessing
        preproc_func_name = self._network_name if self._preproc_func_name is None else self._preproc_func_name
        image_preproc_fn = preprocessing_factory.get_preprocessing(preproc_func_name, is_training=False)
        image_preproc = image_preproc_fn(image, self.image_size, self.image_size)
        # Read a batch of preprocessing images from queue
        image_batch = tf.train.batch(
            [image_preproc, image_filename], batch_size, num_threads=num_threads,
            allow_smaller_final_batch=True)
        return image_batch

    def enqueue_image_files(self, image_files):
        '''
        Given a list of input images, feed these to the queue.
        :param image_files: list of str, list of image files to feed to filename queue
        '''
        self._sess.run(self._enqueue_op, feed_dict={self._pl_image_files: image_files})

    def feed_forward_batch(self, layer_names, images=None, fetch_images=False):
        '''
        Main method for pushing a batch of images through the network. There are
        two input options: (1) feeding a list of image filenames to images or (2)
        using the file input queue. Which input method to use is determined
        by whether the `images` parameter is specified. If None, then the queue
        is used. This function returns a dictionary of outputs in which keys
        correspond to layer names (and 'filenames' and 'examples_in_queue') and
        the tensor values.

        :param layer_names: list of str, layer names to extract features from
        :param images: list of str, optional list of image filenames (default=None)
        :param fetch_images: bool, optionally fetch the input images (default=False)
        :return: dict, dictionary with values for all fetches

        '''

        # Dictionary of network operations (activations) to fetch
        fetches = {}

        # Check if all layers are available
        available_layers = self.layer_names()
        for layer_name in layer_names:
            if layer_name not in available_layers:
                raise ValueError("Unable to extract features for layer: {}".format(layer_name))
            fetches[layer_name] = self._endpoints[layer_name]

        # Manual inputs using placeholder 'images' of shape [N,H,W,C]
        feed_dict = None
        if images is not None:
            feed_dict = {self._image_batch: images}
        else:
            feed_dict = None
            fetches["filenames"] = self._batch_filenames

        # Optionally, we fetch the input image (for debugging/viz)
        if fetch_images:
            fetches["images"] = self._image_batch

        # Fetch how many examples left in queue
        fetches["examples_in_queue"] = self._num_in_queue

        # Actual forward pass through the network
        outputs = self._sess.run(fetches, feed_dict=feed_dict)
        return outputs

    def num_in_queue(self):
        '''
        :return: int, returns the current number of examples in the queue
        '''
        return self._sess.run(self._num_in_queue)

    def layer_names(self):
        '''
        :return: list of str, layer names in the network
        '''
        return self._endpoints.keys()

    def layer_size(self, name):
        '''
        :param name: str, layer name
        :return: list of int, shape of the network layer
        '''
        return self._endpoints[name].get_shape().as_list()

    def print_network_summary(self):
        '''
        Prints the network layers and their shapes
        '''
        for name, tensor in self._endpoints.items():
            print("{} has shape {}".format(name, tensor.shape))

    def close(self):
        '''
        Stop the pre-processing threads and close the session
        '''
        self._coord.request_stop()
        self._sess.run(self._filename_queue.close(cancel_pending_enqueues=True))
        self._coord.join(self._threads)
        self._sess.close()

    @property
    def image_size(self):
        return self._image_size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_preproc_threads(self):
        return self._num_preproc_threads