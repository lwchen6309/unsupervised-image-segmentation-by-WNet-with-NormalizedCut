# unsupervised-image-segmentation-by-WNet-with-NormalizedCut
A tensorflow implementation of [WNet](https://arxiv.org/abs/1711.08506)
for unsupervised image segmentation on VOC2012 dataset

This code is revised from [FCN code by shekkizh](https://github.com/shekkizh/FCN.tensorflow)

# Architecture
![image](https://github.com/lwchen6309/unsupervised-image-segmentation-by-WNet-with-NormalizedCut/blob/master/image/WNet_architecture.png)

Two Unets are stacked as autoencoder to generate sementic segmentation of images.
An additional soft normalized cut serve as penalty term to improve segmentation.

# Prerequisites
The code was tested with tensorflow1.9 and python3.6. 
 
# Results

