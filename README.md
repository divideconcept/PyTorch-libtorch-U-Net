# PyTorch/libtorch Customizable 1D/2D U-Net  
A customizable 1D/2D U-Net model for libtorch (PyTorch c++ UNet)  
Robin Lobel, March 2020 - Requires libtorch 1.4.0 or higher. CPU & CUDA compatible. Qt compatible.

The default parameters produce the original 2D UNet ( https://arxiv.org/pdf/1505.04597.pdf ) with all core improvements activated, resulting in a fully convolutional 2D network.  
The default parameters for the 1D Unet are inspired by the Wave UNet ( https://arxiv.org/pdf/1806.03185.pdf ) with all core improvements activated, resulting in a fully convolutional 1D network.

You can customize the number of in/out channels, the number of hidden feature channels, the number of levels, the size of the kernel, and activate improvements such as:
* Zero-Padding ( Imagenet classification with deep convolutional neural networks, A. Krizhevsky, I. Sutskever, and G. E. Hinton ) which allows the same size for input and output
* Strided Convolution instead of Strided Max Pooling for Downsampling ( https://arxiv.org/pdf/1412.6806.pdf , https://arxiv.org/pdf/1701.03056.pdf , https://arxiv.org/pdf/1606.04797.pdf ) which improve the quality of training and inference
* Resize Convolution instead of Strided Deconvolution for Upsampling ( https://distill.pub/2016/deconv-checkerboard/ , https://www.kaggle.com/mpalermo/remove-grideffect-on-generated-images/notebook , https://arxiv.org/pdf/1806.02658.pdf ) which eliminates checkerboard pattern artefacts
* Partial Convolution ( https://arxiv.org/pdf/1811.11718.pdf , https://github.com/NVIDIA/partialconv ) which correct the errors introduced by zero-padding
* BatchNorm ( https://arxiv.org/abs/1502.03167 ) which can accelerate or stabilize training in some cases

You can additionally display the size of all internal layers the first time you use the network

## How to choose the parameters

* The number of input channels is the number of useful infos you can feed the model for each pixel (for instance 3 channels (RGB) for a picture, 2 channels for a spectrogramme (real/imaginary)).
* The number of output channels is the number of infos you want in the end; it can be the same as the input if you want to get a filtered picture or a spectrogram back, for instance, but can also be any other kind of infos (classification masks...).
* The number of hidden feature channels can only be determined by experimenting (that's why I would recommend to only tweak that parameter last). Start with a low number of feature channels (8 for instance) because the training will go fast, then double it until the output no longer increase in quality (check the loss value, and visualize the results).
* The number of levels can be determined by opening your input samples into a viewer, and then downscale by a factor of 2 several times until you can't discriminate any useful feature anymore. The number of downscales correspond to the number of useful levels for the model.

## Usage (2D UNet)

```c++
#include "cunet.h"

int main(int argc, char *argv[])
{
    int batchSize=16;
    int inChannels=3, outChannels=3;
    int height=512, width=512;
    
    CUNet2d model(inChannels,outChannels);
    torch::optim::Adam optim(model->parameters(), torch::optim::AdamOptions(1e-3));
    
    torch::Tensor source=torch::randn({batchSize,inChannels,height,width});
    torch::Tensor target=torch::randn({batchSize,outChannels,height,width});
    torch::Tensor result, loss;
    
    model->train();
    for (int epoch = 0; epoch < 100; epoch++)
    {
        optim.zero_grad();
        result = model(source);
        loss = torch::mse_loss(result, target);
        loss.backward();
        optim.step();
    }
    
    model->eval();
    torch::Tensor validation=torch::randn({batchSize,inChannels,height,width});
    torch::Tensor inference = model(validation);
    
    return 0;
}
```

## Usage (1D UNet)

```c++
#include "cunet.h"

int main(int argc, char *argv[])
{
    int batchSize=16;
    int inChannels=2, outChannels=2;
    int size=2048;
    
    CUNet1d model(inChannels,outChannels);
    torch::optim::Adam optim(model->parameters(), torch::optim::AdamOptions(1e-3));
    
    torch::Tensor source=torch::randn({batchSize,inChannels,size});
    torch::Tensor target=torch::randn({batchSize,outChannels,size});
    torch::Tensor result, loss;
    
    model->train();
    for (int epoch = 0; epoch < 100; epoch++)
    {
        optim.zero_grad();
        result = model(source);
        loss = torch::mse_loss(result, target);
        loss.backward();
        optim.step();
    }
    
    model->eval();
    torch::Tensor validation=torch::randn({batchSize,inChannels,size});
    torch::Tensor inference = model(validation);
    
    return 0;
}
```
