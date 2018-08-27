# Unsupervised-image-segmentation-by-WNet-with-NormalizedCut
A tensorflow implementation of [WNet](https://arxiv.org/abs/1711.08506)
for unsupervised image segmentation on PASCAL VOC2012 dataset

This code is revised from [FCN code by shekkizh](https://github.com/shekkizh/FCN.tensorflow)

# WNet

![image](https://github.com/lwchen6309/unsupervised-image-segmentation-by-WNet-with-NormalizedCut/blob/master/image/WNet_architecture.png)

(Figure from the original WNet paper)

Two Unets are stacked as autoencoder to generate sementic segmentation of images.

An additional soft normalized cut term serve as a criterion to improve segmentation,

For details of normalized cut, please refer to the [original paper](https://arxiv.org/abs/1711.08506)

In short, normalized cut measure how good the segmentation is, the procedures are as follow:

(1) For a image, we calculate weight connections between any two pixels in image by their 

    (a) Distance of brightness between pixels i.e. (R+G+B)/3.
    
    (b) Distances of position between pixels.
    For pixels with strong connection, they are more likely belong to the same class.

(2) The association and disassociation is then calculated by weight connection:

    (a) association: sum(weight_connection) within the class.
    
    (b) disassociation: sum(weight_connection) between the class.
    
    (c) normalized cut = disassociation / association

# Run program
    (1) The code was tested with tensorflow1.9, python3.6 and a GTX1080 graphic cards. 
    
    (2) Train / visualize network
        (a) Run WNet_naive.py for segmentation without normalized cut.
        (b) Run WNet_bright.py for segmentation with normalized cut.
        
        There are few arguments:
            --mode: [train/visualize/test] 
                train; 
                visualize: randomly select image from dataset and predict segmentation;
                test: load images in test_dir(by default is './data/test') and predict segmentation.
                      test_dir can be also specified by --test_dir.
            --logs_dir: specify the directroy to restore / save.
            --num_layers: specify the number of modules for UNet.
                        (for figure above, the #modules is 9.)
            --debug: [True/False] Print extra details (activations, gradients, ...) for debug.

# Results

Unsupervised image segmentation is perform with and without soft normalized cut.

(1) A 5 modules WNet tested in this work.
(2) The image from VOC2012 is resize to 128 * 128 pixels for limitation of memories.
(3) The Wnet is first train with dropout rate 0.65 for 50000 iterations,
    then retrain with 0.3 for another 50000 iterations.
(5) Learning rate is reduced by half every 10000 iterations.

## (1) WNet naive (without soft normalized cut)
### Training process 
(for simplicity, we show only the fisrt training process.)
![image](https://github.com/lwchen6309/unsupervised-image-segmentation-by-WNet-with-NormalizedCut/blob/master/image/WNet_naive_loss.png)

### Segmentation
![image](https://github.com/lwchen6309/unsupervised-image-segmentation-by-WNet-with-NormalizedCut/blob/master/image/WNet_naive_compare.png)

## (2) WNet bright 
### Training process 
(for simplicity, we show only the fisrt training process.)
![image](https://github.com/lwchen6309/unsupervised-image-segmentation-by-WNet-with-NormalizedCut/blob/master/image/WNet_bright_loss.png)

### Segmentation
![image](https://github.com/lwchen6309/unsupervised-image-segmentation-by-WNet-with-NormalizedCut/blob/master/image/WNet_bright_compare.png)


# Future work
In WNet paper, there is extra post-processing like 
conditional random field (CRF) to acquire satisfactory segmentations,
which is currently not included in this work, and will be added in future.

