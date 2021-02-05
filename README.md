# Unsupervised-image-segmentation-by-WNet-with-NormalizedCut
This is a tensorflow implementation of [WNet](https://arxiv.org/abs/1711.08506)
for unsupervised image segmentation on PASCAL VOC2012 dataset

This source code revised based on [FCN code by shekkizh](https://github.com/shekkizh/FCN.tensorflow)

# WNet

![image](https://github.com/lwchen6309/unsupervised-image-segmentation-by-WNet-with-NormalizedCut/blob/master/image/WNet_architecture.png)

(Figure from the original WNet paper)

The WNet consists on two Unets to stack as autoencoder transforming from image to segmantion map, then back to image.

In ths implementation, an additional soft normalized cut term is added to serve as a criterion to improve the segmentation.

In brief, the soft normalized cut measure how the segmentation map aggregates. Optimizing this measurement helps to remove the spare noise and noisy fragments in the segmentation map. Please refer to the [original paper](https://arxiv.org/abs/1711.08506) for details of normalized cut.

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
    (1) The code was tested based on tensorflow1.9, python3.6 and a GTX1080 graphic card. Please refer the [FCN code by shekkizh](https://github.com/shekkizh/FCN.tensorflow) for further enviroment.
    
    (2) Train / visualize network
        cd to "unsupervised-image-segmentation-by-WNet-with-NormalizedCut"
        (a) Run "python src/WNet_naive.py" for segmentation without normalized cut.
        (b) Run "python src/WNet_bright.py" for segmentation with normalized cut.
        
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
