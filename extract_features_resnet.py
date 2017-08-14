import numpy as np
import os
import tensorflow as tf
import cortex.utils

from datasets import imagenet
from nets import resnet_v2, inception
from preprocessing import inception_preprocessing

from tensorflow.contrib import slim
import matplotlib.pyplot as plt

################################################################################

def input_pipeline(image_path, batch_size, image_size, num_threads=1):

    # Produce a queue with image files to read
    #image_files = tf.train.match_filenames_once(os.path.join(image_path + "*.jpg"))
    #filename_queue = tf.train.string_input_producer(image_files, shuffle=False)

    image_files = cortex.utils.find_files(image_path, ("png", "jpg", "jpeg"))
    image_files_tensor = tf.constant(image_files)
    filename_queue = tf.FIFOQueue(len(image_files), [tf.string], shapes=[[]])
    enqueue_op = filename_queue.enqueue_many([image_files_tensor])

    # Read image file from disk and decode JPEG
    reader = tf.WholeFileReader()
    _, image_raw = reader.read(filename_queue)
    image = tf.image.decode_jpeg(image_raw, channels=3)

    # Input preprocessing
    image_preproc = inception_preprocessing.preprocess_for_eval(image, image_size, image_size)
    return enqueue_op, tf.train.batch([image_preproc], batch_size, num_threads=num_threads)

################################################################################

image_size = inception.inception_v1.default_image_size
#image_size = 224
batch_size = 32

checkpoints_dir = "/home/trunia1/data/MODELS/TensorFlow/ResNet-v2/resnet_v2_101_2017_04_14"
image_path = "/home/trunia1/data/MS-COCO/val2014/"
imagenet_classnames = imagenet.create_readable_names_for_imagenet_labels()

################################################################################

# Queue for loading images from disk and preprocessing them
enqueue_op, batch_images = input_pipeline(image_path, batch_size, image_size)

# Create the model, use the default arg scope to configure the batch norm parameters.
with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    logits, endpoints = resnet_v2.resnet_v2_101(batch_images, num_classes=1001, is_training=False)
    probabilities = tf.nn.softmax(logits)

init_fn = slim.assign_from_checkpoint_fn(
    os.path.join(checkpoints_dir, 'resnet_v2_101.ckpt'),
    slim.get_model_variables('resnet_v2_101'))

# model_variables = slim.get_model_variables('resnet_v2_101')
# for v in model_variables:
#     print(v.name, v.shape)

# Initialize the TF session
sess = tf.Session()
init_fn(sess)

# Local variables initializer
sess.run(tf.local_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

# Add all training examples to the queue
sess.run(enqueue_op)

# for e in endpoints:
#     v = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, e)[0]
#     print(v)

layer_names = ["resnet_v2_101/postnorm/Relu",
               "resnet_v2_101/pool5",
               "resnet_v2_101/SpatialSqueeze"]
layers = [tf.get_default_graph().get_operation_by_name(name) for name in layer_names]

for step in range(1000000):

    if coord.should_stop():
        break

    # Feed-forward through the network
    ops = layer_names
    ops.append(batch_images)
    ops.append(probabilities)

    outputs = sess.run(ops)

    for o in outputs:
        if o is not None:
            print(o.shape)

    # # Show the results
    # for i in range(batch_size):
    #
    #     image = np_images[i]
    #     probability = np_probabilities[i,:]
    #
    #     sorted_inds = [j[0] for j in sorted(enumerate(-probability), key=lambda x:x[1])]
    #     print("Prediction: {} ({:.2f})".format(imagenet_classnames[sorted_inds[0]], probability[sorted_inds[0]]))
    #
    #     plt.figure()
    #     plt.imshow(image.astype(np.uint8))
    #     plt.axis('off')
    #     plt.show()

################################################################################

# Stop the queue threads and properly close the session
coord.request_stop()
coord.join(threads)
sess.close()