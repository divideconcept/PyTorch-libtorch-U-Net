#ifndef CUNET_H
#define CUNET_H

// Customizable 1D/2D UNet for PyTorch C++
// ---------------------------------------------------------
// Robin Lobel, March 2020 - requires libtorch 1.4 and higher - Qt compatible
// https://github.com/divideconcept/PyTorch-libtorch-U-Net
//
// The default parameters produce the original 2D UNet ( https://arxiv.org/pdf/1505.04597.pdf ) with all core improvements activated, resulting in a fully convolutional 2D network
// The default parameters for the 1D Unet are inspired by the Wave UNet ( https://arxiv.org/pdf/1806.03185.pdf ) with all core improvements activated, resulting in a fully convolutional 1D network

// You can customize the number of in/out channels, the number of hidden feature channels, the number of levels, the size of the kernel, and activate improvements such as:
// -Zero-Padding ( Imagenet classification with deep convolutional neural networks, A. Krizhevsky, I. Sutskever, and G. E. Hinton )
// -Strided Convolution instead of Strided Max Pooling for Downsampling ( https://arxiv.org/pdf/1412.6806.pdf, https://arxiv.org/pdf/1701.03056.pdf , https://arxiv.org/pdf/1606.04797.pdf )
// -Resize Convolution instead of Strided Deconvolution for Upsampling ( https://distill.pub/2016/deconv-checkerboard/ , https://www.kaggle.com/mpalermo/remove-grideffect-on-generated-images/notebook , https://arxiv.org/pdf/1806.02658.pdf )
// -Partial Convolution to fix Zero-Padding bias ( https://arxiv.org/pdf/1811.11718.pdf , https://github.com/NVIDIA/partialconv )
// -BatchNorm ( https://arxiv.org/abs/1502.03167 )
// You can additionally display the size of all internal layers the first time you call forward()

//Qt compatibility
#ifdef QT_CORE_LIB
    #undef slots
#endif
#include<torch/torch.h>
#ifdef QT_CORE_LIB
    #define slots Q_SLOTS
#endif

#include "partialconv.h"

struct CUNet2dImpl : torch::nn::Module {
    CUNet2dImpl(int32_t inChannels, int32_t outChannels, int32_t featureChannels=64, int32_t levels=5, int32_t kernelSize=3, bool padding=true, bool convolutionDownsampling=true, bool convolutionUpsampling=true, bool partialConvolution=true, bool batchNorm=false, bool showSizes=false)
    {
        this->levels=levels;
        this->kernelSize=kernelSize;
        this->paddingSize=padding?(kernelSize-1)/2:0;

        this->convolutionDownsampling=convolutionDownsampling;
        this->convolutionUpsampling=convolutionUpsampling;
        this->partialConvolution=partialConvolution;
        this->batchNorm=batchNorm;
        this->showSizes=showSizes;

        for(int level=0; level<levels-1; level++)
        {
            contracting.push_back(levelBlock(level==0?inChannels:featureChannels*(1<<(level-1)), featureChannels*(1<<level)));
            register_module("contractingBlock"+std::to_string(level),contracting.back());

            downsampling.push_back(downsamplingBlock(featureChannels*(1<<level)));
            register_module("downsampling"+std::to_string(level),downsampling.back());
        }

        bottleneck=levelBlock(featureChannels*(1<<(levels-2)), featureChannels*(1<<(levels-1)));
        register_module("bottleneck",bottleneck);

        for(int level=levels-2; level>=0; level--)
        {
            upsampling.push_front(upsamplingBlock(featureChannels*(1<<(level+1)), featureChannels*(1<<level)));
            register_module("upsampling"+std::to_string(level),upsampling.front());

            expanding.push_front(levelBlock(featureChannels*(1<<level)*2, featureChannels*(1<<level)));
            register_module("expandingBlock"+std::to_string(level),expanding.front());
        }

        output=torch::nn::Conv2d(torch::nn::Conv2dOptions(featureChannels, outChannels, 1));
        register_module("output",output);
    }

    torch::Tensor forward(const torch::Tensor& inputTensor) {
        std::vector<torch::Tensor> contractingTensor(levels-1);
        std::vector<torch::Tensor> downsamplingTensor(levels-1);
        torch::Tensor bottleneckTensor;
        std::vector<torch::Tensor> upsamplingTensor(levels-1);
        std::vector<torch::Tensor> expandingTensor(levels-1);
        torch::Tensor outputTensor;

        for(int level=0; level<levels-1; level++)
        {
            contractingTensor[level]=contracting[level]->forward(level==0?inputTensor:downsamplingTensor[level-1]);
            downsamplingTensor[level]=downsampling[level]->forward(contractingTensor[level]);
        }

        bottleneckTensor=bottleneck->forward(downsamplingTensor.back());

        for(int level=levels-2; level>=0; level--)
        {
            upsamplingTensor[level]=upsampling[level]->forward(level==levels-2?bottleneckTensor:expandingTensor[level+1]);
            if(paddingSize==0) { //apply cropping to the contracting tensor in order to concatenate with the same-level expanding tensor
                int oldXSize=contractingTensor[level].size(2);
                int oldYSize=contractingTensor[level].size(3);
                int newXSize=upsamplingTensor[level].size(2);
                int newYSize=upsamplingTensor[level].size(3);
                int startX=oldXSize/2-newXSize/2;
                int startY=oldYSize/2-newYSize/2;
                contractingTensor[level]=contractingTensor[level].slice(2,startX,startX+newXSize);
                contractingTensor[level]=contractingTensor[level].slice(3,startY,startY+newYSize);
            }
            expandingTensor[level]=expanding[level]->forward(torch::cat({contractingTensor[level],upsamplingTensor[level]},1));
        }

        outputTensor=output->forward(expandingTensor.front());

        if(showSizes)
        {
            std::cout << "input:  " << inputTensor.sizes() << std::endl;
            for(int level=0; level<levels-1; level++)
            {
                for(int i=0; i<level; i++) std::cout << " "; std::cout << " contracting" << level << ":  " << contractingTensor[level].sizes() << std::endl;
                for(int i=0; i<level; i++) std::cout << " "; std::cout << " downsampling" << level << ": " << downsamplingTensor[level].sizes() << std::endl;
            }
            for(int i=0; i<levels-1; i++) std::cout << " "; std::cout << " bottleneck:    " << bottleneckTensor.sizes() << std::endl;
            for(int level=levels-2; level>=0; level--)
            {
                for(int i=0; i<level; i++) std::cout << " "; std::cout << " upsampling" << level << ":   " << upsamplingTensor[level].sizes() << std::endl;
                for(int i=0; i<level; i++) std::cout << " "; std::cout << " expanding" << level << ":    " << expandingTensor[level].sizes() << std::endl;
            }
            std::cout << "output: " << outputTensor.sizes() << std::endl;
            showSizes=false;
        }

        return outputTensor;
    }

    //the 2d tensor size you pass to the model must be a multiple of this
    int sizeMultiple() {return 1<<(levels-1);}
private:
    torch::nn::Sequential levelBlock(int inChannels, int outChannels)
    {
        if(batchNorm)
        {
            if(partialConvolution)
                return torch::nn::Sequential(
                            PartialConv2d(torch::nn::Conv2dOptions(inChannels, outChannels, kernelSize).padding(paddingSize)),
                            torch::nn::BatchNorm2d(outChannels),
                            torch::nn::ReLU(),
                            PartialConv2d(torch::nn::Conv2dOptions(outChannels, outChannels, kernelSize).padding(paddingSize)),
                            torch::nn::BatchNorm2d(outChannels),
                            torch::nn::ReLU()
                        );
            else
                return torch::nn::Sequential(
                            torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels, outChannels, kernelSize).padding(paddingSize)),
                            torch::nn::BatchNorm2d(outChannels),
                            torch::nn::ReLU(),
                            torch::nn::Conv2d(torch::nn::Conv2dOptions(outChannels, outChannels, kernelSize).padding(paddingSize)),
                            torch::nn::BatchNorm2d(outChannels),
                            torch::nn::ReLU()
                        );
        } else {
            if(partialConvolution)
                return torch::nn::Sequential(
                            PartialConv2d(torch::nn::Conv2dOptions(inChannels, outChannels, kernelSize).padding(paddingSize)),
                            torch::nn::ReLU(),
                            PartialConv2d(torch::nn::Conv2dOptions(outChannels, outChannels, kernelSize).padding(paddingSize)),
                            torch::nn::ReLU()
                        );
            else
                return torch::nn::Sequential(
                            torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels, outChannels, kernelSize).padding(paddingSize)),
                            torch::nn::ReLU(),
                            torch::nn::Conv2d(torch::nn::Conv2dOptions(outChannels, outChannels, kernelSize).padding(paddingSize)),
                            torch::nn::ReLU()
                        );
        }
    }

    torch::nn::Sequential downsamplingBlock(int channels)
    {
        if(convolutionDownsampling)
        {
            if(partialConvolution)
                return torch::nn::Sequential(
                            PartialConv2d(torch::nn::Conv2dOptions(channels, channels, kernelSize).stride(2).padding(paddingSize))
                        );
            else
                return torch::nn::Sequential(
                            torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, kernelSize).stride(2).padding(paddingSize))
                        );
        } else {
            return torch::nn::Sequential(
                        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
                    );
        }
    }

    torch::nn::Sequential upsamplingBlock(int inChannels, int outChannels)
    {
        if(convolutionUpsampling)
        {
            if(partialConvolution)
                return torch::nn::Sequential(
                            torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({2, 2})).mode(torch::kNearest)),
                            PartialConv2d(torch::nn::Conv2dOptions(inChannels, outChannels, kernelSize).padding(paddingSize))
                        );
            else
                return torch::nn::Sequential(
                            torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({2, 2})).mode(torch::kNearest)),
                            torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels, outChannels, kernelSize).padding(paddingSize))
                        );
        } else {
            return torch::nn::Sequential(
                        torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(inChannels, outChannels, 2).stride(2))
                    );
        }
    }

    int levels;
    int kernelSize;
    int paddingSize;

    bool convolutionDownsampling;
    bool convolutionUpsampling;
    bool partialConvolution;
    bool batchNorm;
    bool showSizes;

    std::deque<torch::nn::Sequential> contracting;
    std::deque<torch::nn::Sequential> downsampling;
    torch::nn::Sequential bottleneck;
    std::deque<torch::nn::Sequential> upsampling;
    std::deque<torch::nn::Sequential> expanding;
    torch::nn::Conv2d output{nullptr};
};
TORCH_MODULE_IMPL(CUNet2d, CUNet2dImpl);



struct CUNet1dImpl : torch::nn::Module {
    CUNet1dImpl(int32_t inChannels, int32_t outChannels, int32_t featureChannels=24, int32_t levels=12, int32_t kernelSize=5, bool padding=true, bool convolutionDownsampling=true, bool convolutionUpsampling=true, bool partialConvolution=true, bool batchNorm=false, bool showSizes=false)
    {
        this->levels=levels;
        this->kernelSize=kernelSize;
        this->paddingSize=padding?(kernelSize-1)/2:0;

        this->convolutionDownsampling=convolutionDownsampling;
        this->convolutionUpsampling=convolutionUpsampling;
        this->partialConvolution=partialConvolution;
        this->batchNorm=batchNorm;
        this->showSizes=showSizes;

        for(int level=0; level<levels-1; level++)
        {
            contracting.push_back(levelBlock(level==0?inChannels:featureChannels*(level), featureChannels*(level+1)));
            register_module("contractingBlock"+std::to_string(level),contracting.back());

            downsampling.push_back(downsamplingBlock(featureChannels*(level+1)));
            register_module("downsampling"+std::to_string(level),downsampling.back());
        }

        bottleneck=levelBlock(featureChannels*(levels-1), featureChannels*(levels));
        register_module("bottleneck",bottleneck);

        for(int level=levels-2; level>=0; level--)
        {
            upsampling.push_front(upsamplingBlock(featureChannels*(level+2), featureChannels*(level+1)));
            register_module("upsampling"+std::to_string(level),upsampling.front());

            expanding.push_front(levelBlock(featureChannels*(level+1)*2, featureChannels*(level+1)));
            register_module("expandingBlock"+std::to_string(level),expanding.front());
        }

        output=torch::nn::Conv1d(torch::nn::Conv1dOptions(featureChannels, outChannels, 1));
        register_module("output",output);
    }

    torch::Tensor forward(const torch::Tensor& inputTensor) {
        std::vector<torch::Tensor> contractingTensor(levels-1);
        std::vector<torch::Tensor> downsamplingTensor(levels-1);
        torch::Tensor bottleneckTensor;
        std::vector<torch::Tensor> upsamplingTensor(levels-1);
        std::vector<torch::Tensor> expandingTensor(levels-1);
        torch::Tensor outputTensor;

        for(int level=0; level<levels-1; level++)
        {
            contractingTensor[level]=contracting[level]->forward(level==0?inputTensor:downsamplingTensor[level-1]);
            downsamplingTensor[level]=downsampling[level]->forward(contractingTensor[level]);
        }

        bottleneckTensor=bottleneck->forward(downsamplingTensor.back());

        for(int level=levels-2; level>=0; level--)
        {
            upsamplingTensor[level]=upsampling[level]->forward(level==levels-2?bottleneckTensor:expandingTensor[level+1]);
            if(paddingSize==0) { //apply cropping to the contracting tensor in order to concatenate with the same-level expanding tensor
                int oldXSize=contractingTensor[level].size(2);
                int newXSize=upsamplingTensor[level].size(2);
                int startX=oldXSize/2-newXSize/2;
                contractingTensor[level]=contractingTensor[level].slice(2,startX,startX+newXSize);
            }
            expandingTensor[level]=expanding[level]->forward(torch::cat({contractingTensor[level],upsamplingTensor[level]},1));
        }

        outputTensor=output->forward(expandingTensor.front());

        if(showSizes)
        {
            std::cout << "input:  " << inputTensor.sizes() << std::endl;
            for(int level=0; level<levels-1; level++)
            {
                for(int i=0; i<level; i++) std::cout << " "; std::cout << " contracting" << level << ":  " << contractingTensor[level].sizes() << std::endl;
                for(int i=0; i<level; i++) std::cout << " "; std::cout << " downsampling" << level << ": " << downsamplingTensor[level].sizes() << std::endl;
            }
            for(int i=0; i<levels-1; i++) std::cout << " "; std::cout << " bottleneck:    " << bottleneckTensor.sizes() << std::endl;
            for(int level=levels-2; level>=0; level--)
            {
                for(int i=0; i<level; i++) std::cout << " "; std::cout << " upsampling" << level << ":   " << upsamplingTensor[level].sizes() << std::endl;
                for(int i=0; i<level; i++) std::cout << " "; std::cout << " expanding" << level << ":    " << expandingTensor[level].sizes() << std::endl;
            }
            std::cout << "output: " << outputTensor.sizes() << std::endl;
            showSizes=false;
        }

        return outputTensor;
    }

    //the 1d tensor size you pass to the model must be a multiple of this
    int sizeMultiple() {return 1<<(levels-1);}
private:
    torch::nn::Sequential levelBlock(int inChannels, int outChannels)
    {
        if(batchNorm)
        {
            if(partialConvolution)
                return torch::nn::Sequential(
                            PartialConv1d(torch::nn::Conv1dOptions(inChannels, outChannels, kernelSize).padding(paddingSize)),
                            torch::nn::BatchNorm1d(outChannels),
                            torch::nn::ReLU(),
                            PartialConv1d(torch::nn::Conv1dOptions(outChannels, outChannels, kernelSize).padding(paddingSize)),
                            torch::nn::BatchNorm1d(outChannels),
                            torch::nn::ReLU()
                        );
            else
                return torch::nn::Sequential(
                            torch::nn::Conv1d(torch::nn::Conv1dOptions(inChannels, outChannels, kernelSize).padding(paddingSize)),
                            torch::nn::BatchNorm1d(outChannels),
                            torch::nn::ReLU(),
                            torch::nn::Conv1d(torch::nn::Conv1dOptions(outChannels, outChannels, kernelSize).padding(paddingSize)),
                            torch::nn::BatchNorm1d(outChannels),
                            torch::nn::ReLU()
                        );
        } else {
            if(partialConvolution)
                return torch::nn::Sequential(
                            PartialConv1d(torch::nn::Conv1dOptions(inChannels, outChannels, kernelSize).padding(paddingSize)),
                            torch::nn::ReLU(),
                            PartialConv1d(torch::nn::Conv1dOptions(outChannels, outChannels, kernelSize).padding(paddingSize)),
                            torch::nn::ReLU()
                        );
            else
                return torch::nn::Sequential(
                            torch::nn::Conv1d(torch::nn::Conv1dOptions(inChannels, outChannels, kernelSize).padding(paddingSize)),
                            torch::nn::ReLU(),
                            torch::nn::Conv1d(torch::nn::Conv1dOptions(outChannels, outChannels, kernelSize).padding(paddingSize)),
                            torch::nn::ReLU()
                        );
        }
    }

    torch::nn::Sequential downsamplingBlock(int channels)
    {
        if(convolutionDownsampling)
        {
            if(partialConvolution)
                return torch::nn::Sequential(
                            PartialConv1d(torch::nn::Conv1dOptions(channels, channels, kernelSize).stride(2).padding(paddingSize))
                        );
            else
                return torch::nn::Sequential(
                            torch::nn::Conv1d(torch::nn::Conv1dOptions(channels, channels, kernelSize).stride(2).padding(paddingSize))
                        );
        } else {
            return torch::nn::Sequential(
                        torch::nn::MaxPool1d(torch::nn::MaxPool1dOptions(2).stride(2))
                    );
        }
    }

    torch::nn::Sequential upsamplingBlock(int inChannels, int outChannels)
    {
        if(convolutionUpsampling)
        {
            if(partialConvolution)
                return torch::nn::Sequential(
                            torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({2})).mode(torch::kNearest)),
                            PartialConv1d(torch::nn::Conv1dOptions(inChannels, outChannels, kernelSize).padding(paddingSize))
                        );
            else
                return torch::nn::Sequential(
                            torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({2})).mode(torch::kNearest)),
                            torch::nn::Conv1d(torch::nn::Conv1dOptions(inChannels, outChannels, kernelSize).padding(paddingSize))
                        );
        } else {
            return torch::nn::Sequential(
                        torch::nn::ConvTranspose1d(torch::nn::ConvTranspose1dOptions(inChannels, outChannels, 2).stride(2))
                    );
        }
    }

    int levels;
    int kernelSize;
    int paddingSize;

    bool convolutionDownsampling;
    bool convolutionUpsampling;
    bool partialConvolution;
    bool batchNorm;
    bool showSizes;

    std::deque<torch::nn::Sequential> contracting;
    std::deque<torch::nn::Sequential> downsampling;
    torch::nn::Sequential bottleneck;
    std::deque<torch::nn::Sequential> upsampling;
    std::deque<torch::nn::Sequential> expanding;
    torch::nn::Conv1d output{nullptr};
};
TORCH_MODULE_IMPL(CUNet1d, CUNet1dImpl);

#endif // CUNET_H
