# TensorFlow Feature Extractor

This is a convenient wrapper for **feature extraction** or **classification** in TensorFlow. Given well known pre-trained models on ImageNet, the extractor runs over a list or directory of images. Optionally, features can be saved as HDF5 file. It supports all the [pre-trained models](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models) listed on the official page.

**TensorFlow models tested:**

1. Inception v1-v4
2. ResNet v1 and v2
3. VGG 16-19

## Requirements

* [TensorFlow](https://github.com/tensorflow) (tested with version 1.8)
* [TensorFlow Models](https://github.com/tensorflow/models/)
* The usual suspects: `numpy`, `scipy`. 
* Optionally `h5py` for saving features to HDF5 file


## Setup

1. Checkout the TensorFlow `models` repository somewhere on your machine. The path where you checkout the repository will be denoted `<checkout_dir>/models`

```
git clone https://github.com/tensorflow/models/
```  

2. Add the directory `<checkout_dir>/research/slim` to the`$PYTHONPATH` variable. Or add a line to your `.bashrc` file.

```
export PYTHONPATH="<checkout_dir>/research/slim:$PYTHONPATH"
```

3. Download the model checkpoints from the [official page](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models).

## Usage

There are two example files, one for classification and one for feature extraction.

### Feature Extraction

**ResNet-v1-101**
```
example_feat_extract.py 
--network resnet_v1_101 
--checkpoint ./checkpoints/resnet_v1_101.ckpt 
--image_path ./images_dir/ 
--out_file ./features.h5
--num_classes 1000 
--layer_names resnet_v1_101/logits
```

**ResNet-v2-101**
```
example_feat_extract.py 
--network resnet_v2_101 
--checkpoint ./checkpoints/resnet_v2_101.ckpt 
--image_path ./images_dir/
--out_file ./features.h5 
--layer_names resnet_v2_101/logits 
--preproc_func inception
```

**Inception-v4**
```
example_feat_extract.py 
--network inception_v4 
--checkpoint ./checkpoints/inception_v4.ckpt 
--image_path ./images_dir/
--out_file ./features.h5 
--layer_names Logits
```

### Image Classification

```
example_classification.py
--network resnet_v1_101 
--checkpoint ./checkpoints/resnet_v1_101.ckpt 
--image_path ./images_dir/
--num_classes 1000 
--logits_name resnet_v1_101/logits
```


## Work in Progress

1. ~~Save image file names to HDF5 file~~
2. ~~Support for multi-threaded preprocessing~~
