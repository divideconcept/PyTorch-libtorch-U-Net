#ifndef CUNET_H
#define CUNET_H

// Customizable UNet for PyTorch C++
// ---------------------------------------------------------
// Robin Lobel, March 2020 - requires libtorch 1.4 and higher - Qt compatible
// https://github.com/divideconcept/PyTorch-libtorch-U-Net
//
// The default parameters produce the original UNet ( https://arxiv.org/pdf/1505.04597.pdf ) with all improvements activated
// You can customize the number of in/out channels, the number of hidden feature channels, the number of levels, and activate improvements such as:
// -Zero-Padding ( Imagenet classification with deep convolutional neural networks, A. Krizhevsky, I. Sutskever, and G. E. Hinton ),
// -BatchNorm after ReLU ( https://arxiv.org/abs/1502.03167 , https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md ),
// -Strided Convolution instead of Strided Max Pooling for Downsampling ( https://arxiv.org/pdf/1701.03056.pdf , https://arxiv.org/pdf/1412.6806.pdf , https://arxiv.org/pdf/1606.04797.pdf ),
// -Resize Convolution instead of Strided Deconvolution for Upsampling ( https://distill.pub/2016/deconv-checkerboard/ , https://www.kaggle.com/mpalermo/remove-grideffect-on-generated-images/notebook , https://arxiv.org/pdf/1806.02658.pdf )
// You can additionally display the size of all internal layers the first time you call forward()

//Qt compatibility
#ifdef QT_CORE_LIB
    #undef slots
#endif
#include<torch/torch.h>
#ifdef QT_CORE_LIB
    #define slots Q_SLOTS
#endif

struct CUNetImpl : torch::nn::Module {
    CUNetImpl(int32_t inChannels=1, int32_t outChannels=1, int32_t featureChannels=64, int32_t levels=4, bool padding=true, bool batchNorm=true, bool convolutionDownsampling=true, bool convolutionUpsampling=true, bool showSizes=false)
    {
        this->levels=levels;
        paddingSize=padding?1:0;
        this->showSizes=showSizes;

        for(int level=0; level<levels; level++)
        {
            contracting.push_back(levelBlock(level==0?inChannels:featureChannels*(1<<(level-1)), featureChannels*(1<<level), paddingSize, batchNorm));
            register_module("contractingBlock"+std::to_string(level),contracting.back());

            downsampling.push_back(downsamplingBlock(featureChannels*(1<<level),convolutionDownsampling));
            register_module("downsampling"+std::to_string(level),downsampling.back());
        }

        bottleneck=levelBlock(featureChannels*(1<<(levels-1)), featureChannels*(1<<levels), paddingSize, batchNorm);
        register_module("bottleneck",bottleneck);

        for(int level=levels-1; level>=0; level--)
        {
            upsampling.push_front(upsamplingBlock(featureChannels*(1<<(level+1)), featureChannels*(1<<level), convolutionUpsampling));
            register_module("upsampling"+std::to_string(level),upsampling.front());

            expanding.push_front(levelBlock(featureChannels*(1<<(level+1)), featureChannels*(1<<level), paddingSize, batchNorm));
            register_module("expandingBlock"+std::to_string(level),expanding.front());
        }

        output=torch::nn::Conv2d(torch::nn::Conv2dOptions(featureChannels, outChannels, 1));
        register_module("output",output);
    }

    torch::Tensor forward(const torch::Tensor& inputTensor) {
        std::vector<torch::Tensor> contractingTensor(levels);
        std::vector<torch::Tensor> downsamplingTensor(levels);
        torch::Tensor bottleneckTensor;
        std::vector<torch::Tensor> upsamplingTensor(levels);
        std::vector<torch::Tensor> expandingTensor(levels);
        torch::Tensor outputTensor;

        for(int level=0; level<levels; level++)
        {
            contractingTensor[level]=contracting[level]->forward(level==0?inputTensor:downsamplingTensor[level-1]);
            downsamplingTensor[level]=downsampling[level]->forward(contractingTensor[level]);
        }

        bottleneckTensor=bottleneck->forward(downsamplingTensor.back());

        for(int level=levels-1; level>=0; level--)
        {
            upsamplingTensor[level]=upsampling[level]->forward(level==levels-1?bottleneckTensor:expandingTensor[level+1]);
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
            for(int level=0; level<levels; level++)
            {
                for(int i=0; i<level; i++) std::cout << " "; std::cout << " contracting" << level << ":  " << contractingTensor[level].sizes() << std::endl;
                for(int i=0; i<level; i++) std::cout << " "; std::cout << " downsampling" << level << ": " << downsamplingTensor[level].sizes() << std::endl;
            }
            for(int i=0; i<levels; i++) std::cout << " "; std::cout << " bottleneck:    " << bottleneckTensor.sizes() << std::endl;
            for(int level=levels-1; level>=0; level--)
            {
                for(int i=0; i<level; i++) std::cout << " "; std::cout << " upsampling" << level << ":  " << upsamplingTensor[level].sizes() << std::endl;
                for(int i=0; i<level; i++) std::cout << " "; std::cout << " expanding" << level << ":   " << expandingTensor[level].sizes() << std::endl;
            }
            std::cout << "output: " << outputTensor.sizes() << std::endl;
            showSizes=false;
        }

        return outputTensor;
    }

    //the 2d size you pass to the model must be a multiple of this
    int sizeMultiple2d() {return 1<<levels;}
private:
    torch::nn::Sequential levelBlock(int inChannels, int outChannels, int paddingSize, bool batchNorm)
    {
        if(batchNorm)
        {
            return torch::nn::Sequential(
                        torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels, outChannels, 3).padding(paddingSize)),
                        torch::nn::ReLU(),
                        torch::nn::BatchNorm2d(outChannels),
                        torch::nn::Conv2d(torch::nn::Conv2dOptions(outChannels, outChannels, 3).padding(paddingSize)),
                        torch::nn::ReLU(),
                        torch::nn::BatchNorm2d(outChannels)
                    );
        } else {
            return torch::nn::Sequential(
                        torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels, outChannels, 3).padding(paddingSize)),
                        torch::nn::ReLU(),
                        torch::nn::Conv2d(torch::nn::Conv2dOptions(outChannels, outChannels, 3).padding(paddingSize)),
                        torch::nn::ReLU()
                    );
        }
    }

    torch::nn::Sequential downsamplingBlock(int channels, bool convolutionDownsampling)
    {
        if(convolutionDownsampling)
        {
            return torch::nn::Sequential(
                        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 2).stride(2))
                    );
        } else {
            return torch::nn::Sequential(
                        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
                    );
        }
    }

    torch::nn::Sequential upsamplingBlock(int inChannels, int outChannels, bool convolutionUpsampling)
    {
        if(convolutionUpsampling)
        {
            return torch::nn::Sequential(
                        torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor({2, 2}).mode(torch::kNearest)),
                        torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels, outChannels, 3).padding(1))
                    );
        } else {
            return torch::nn::Sequential(
                        torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(inChannels, outChannels, 2).stride(2))
                    );
        }
    }

    int levels;
    int paddingSize;
    bool showSizes;

    std::deque<torch::nn::Sequential> contracting;
    std::deque<torch::nn::Sequential> downsampling;
    torch::nn::Sequential bottleneck;
    std::deque<torch::nn::Sequential> upsampling;
    std::deque<torch::nn::Sequential> expanding;
    torch::nn::Conv2d output{nullptr};
};
TORCH_MODULE_IMPL(CUNet, CUNetImpl);

#endif // CUNET_H
