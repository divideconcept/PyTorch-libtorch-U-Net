# PyTorch-libtorch-U-Net
A customizable U-Net model for PyTorch (libtorch c++ UNet)
Requires libtorch 1.4.0 or higher.

The default parameters of the constructor creates the original U-Net from the paper U-Net: Convolutional Networks for Biomedical Image Segmentation but you can customize the number of in/out channels, the number of hidden feature channels, the number of levels (depth) and the dropout probability. Additionally, you can also display the sizes of the different layers within the model.
