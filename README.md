# Under processing ...



# Unsupervised-image-segmentation-by-WNet-with-NormalizedCut
A tensorflow implementation of [WNet](https://arxiv.org/abs/1711.08506)
for unsupervised image segmentation on VOC2012 dataset

This code is revised from [FCN code by shekkizh](https://github.com/shekkizh/FCN.tensorflow)

# WNet
The following figure from the original paper:
![image](https://github.com/lwchen6309/unsupervised-image-segmentation-by-WNet-with-NormalizedCut/blob/master/image/WNet_architecture.png)

Two Unets are stacked as autoencoder to generate sementic segmentation of images.

An additional soft normalized cut term serve as a criterion to improve segmentation,

For details of normalized cut, please refer to the [original paper](https://arxiv.org/abs/1711.08506)

In short, normalized cut measure how good the segmentation is, the procedures are as follow:

    (1) For a image, we calculate weight connections between any two pixels in image by their 

        brightness of pixels i.e. (R+G+B)/3.
        distances.

    (2) The association and disassociation is then calculated by weight connection.

    (3) The normalized cut is then calculated by: 

        Total normalized disassociation between the groups.
        Total normalized association within the groups.

# Run program
    (1) The code was tested with tensorflow1.9 and python3.6. 
    
    (2) Train / visualize network
        (a) Run WNet_naive.py for segmentation without normalized cut.
        (b) Run WNet_bright.py for segmentation with normalized cut.
        
        There are few arguments:
            --num_layers: specify the number of modules for UNet.
                        (for figure above, the #module is 9.)
            --debug: [True/False] Print extra details (activations, gradients, ...) for debug.
            --mode: [train/visualize/test] 
                    train; 
                    visualize: randomly select image from dataset and predict segmentation.
                    test: load images and predict segmentation.
    
# Results

Unsupervised image segmentation is perform with and without soft normalized cut.

## (1) WNet naive (without soft normalized cut)
To be continued ...

## (2) WNet bright 
To be continued ...

