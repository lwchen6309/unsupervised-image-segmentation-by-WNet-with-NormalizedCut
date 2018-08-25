# unsupervised-image-segmentation-by-WNet-with-NormalizedCut
A tensorflow implementation of [WNet](https://arxiv.org/abs/1711.08506)
for unsupervised image segmentation on VOC2012 dataset

This code is revised from [FCN code by shekkizh](https://github.com/shekkizh/FCN.tensorflow)

# Introduction
## WNet
The following figure from the original paper:
![image](https://github.com/lwchen6309/unsupervised-image-segmentation-by-WNet-with-NormalizedCut/blob/master/image/WNet_architecture.png)

Two Unets are stacked as autoencoder to generate sementic segmentation of images.

An additional soft normalized cut serve as penalty term to improve segmentation.

## Normalized cut



# Run program
    The code was tested with tensorflow1.9 and python3.6. 
    
    (2) Run time_resolved_spectra_NMF.py 
 
 
# Results

Unsupervised image segmentation is perform with and without soft normalized cut.

## (1) WNet naive (without soft normalized cut)

## (2) WNet bright (with soft normalized cut that weight connection is defined on brightness of pixels.)
