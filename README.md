# Image Colorization using CNN
This repository contains a image colorization system using Convolutional Neural nets. The fundamental idea is to predict A and B channels of LAB space images provided the L channels. CIE L*a*b* or sometimes abbreviated as simply "Lab" color space expresses color as three numerical values, L* for the lightness and a* and b* for the green–red and blue–yellow color components. For further details of the color space kindly refer to the following link:

https://en.wikipedia.org/wiki/CIELAB_color_space

The architechture of the network is given by the following

![alt text](https://github.com/Arghyadeep/Image-Colorization-using-CNN/blob/master/process.png)

The idea is inspired by Richard Zhang's image colorization paper: https://arxiv.org/pdf/1603.08511
But instead of upsampling and finding A and B values of the predicted channels from a probability distribution of 313 values as mentioned in the paper, a simple square loss is used to predict. Though simple to implement, a downside of this Loss function is that the images loses its vibrancy in many cases. 

