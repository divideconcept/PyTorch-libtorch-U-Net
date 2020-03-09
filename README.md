# PyTorch/libtorch Customizable U-Net  
A customizable U-Net model for libtorch (PyTorch c++ UNet)  
Robin Lobel, March 2020 - Requires libtorch 1.4.0 or higher. Qt compatible.

The default parameters produce the original UNet ( https://arxiv.org/pdf/1505.04597.pdf )
You can customize the number of in/out channels, the number of hidden feature channels, the number of levels, and activate improvements such as:
* Zero-Padding ( Imagenet classification with deep convolutional neural networks, A. Krizhevsky, I. Sutskever, and G. E. Hinton ),
* BatchNorm after ReLU ( https://arxiv.org/abs/1502.03167 , https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md ),
* Strided Convolution instead of Strided Max Pooling for Downsampling ( https://arxiv.org/pdf/1701.03056.pdf , https://arxiv.org/pdf/1412.6806.pdf , https://arxiv.org/pdf/1606.04797.pdf ),
* Resize Convolution instead of Strided Deconvolution for Upsampling ( https://distill.pub/2016/deconv-checkerboard/ , https://www.kaggle.com/mpalermo/remove-grideffect-on-generated-images/notebook , https://arxiv.org/pdf/1806.02658.pdf )

You can additionally display the size of all internal layers the first time you call forward()

## How to choose the parameters

* The number of input channels is the number of useful infos you can feed the model for each pixel (for instance 3 channels (RGB) for a picture, 2 channels for a spectrogramme (real/imaginary)).
* The number of output channels is the number of infos you want in the end; it can be the same as the input if you want to get a filtered picture or a spectrogram back, for instance, but can also be any other kind of infos (classification masks...).
* The number of hidden feature channels can only be determined by experimenting (that's why I would recommend to only tweak that parameter last). Start with a low number of feature channels (8 for instance) because the training will go fast, then double it until the output no longer increase in quality (check the loss value, and visualize the results).
* The number of levels can be determined by opening your input samples into a viewer, and then downscale by a factor of 2 several times until you can't discriminate any useful feature anymore. The number of downscales correspond to the number of useful levels for the model.
* Make sure each sample is large enough so that zero-padding (if you use it) does not influence the kernel weights too much - which would impact the quality of the training. For instance for a 2 levels UNet, the size of your samples should be 64x64 minimum (double width and height for each additionnal level).
