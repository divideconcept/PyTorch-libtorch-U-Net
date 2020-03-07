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
