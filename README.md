# TensorFlow Feature Extractor

This is a convenient wrapper for feature extraction in TensorFlow. It supports all the pre-trained models on the [tensorflow/models](https://github.com/tensorflow/models/tree/master/slim#pre-trained-models) page. Some of the networks supported are:

* Inception v1-v4
* ResNet v1 and v2
* VGG 16-19


## Setup

1. Checkout the TensorFlow `models` repository somewhere on your machine. The path where you checkout the repository will be denoted `<checkout_dir>/models`

```
git clone https://github.com/tensorflow/models/
```  

2. Add the directory `<checkout_dir>/models` to the`$PYTHONPATH` variable. Or add a line to your `.bashrc` file.

```
export PYTHONPATH="<checkout_dir>/models:$PYTHONPATH"
```

## Examples

**ResNet-v1-101**
```
example.py --network resnet_v1_101 --checkpoint /home/trunia1/data/MODELS/TensorFlow/ResNet/resnet_v1_101.ckpt --image_path /home/trunia1/data/SOS/img_pascal/ --num_classes 1000 --layer_names resnet_v1_101/logits
```

**ResNet-v2-101**
```
example.py --network resnet_v2_101 --checkpoint /home/trunia1/data/MODELS/TensorFlow/ResNet-v2/resnet_v2_101_2017_04_14/resnet_v2_101.ckpt --image_path /home/trunia1/data/SOS/img_pascal/ --layer_names resnet_v2_101/logits --preproc_func inception
```

**Inception-v4**
```
example.py --network inception_v4 --checkpoint /home/trunia1/data/MODELS/TensorFlow/Inception/inception_v4.ckpt --image_path /home/trunia1/data/SOS/img_pascal/ --layer_names Logits
```