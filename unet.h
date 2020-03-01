#ifndef UNET_H
#define UNET_H

#ifdef QT_CORE_LIB
    #undef slots
#endif
#include<torch/torch.h>
#ifdef QT_CORE_LIB
    #define slots Q_SLOTS
#endif

using namespace torch;
using namespace std;

struct UNetImpl : nn::Module {
    //constructor defaults is the original UNet
    UNetImpl(int32_t inChannels=1, int32_t outChannels=1, int32_t featureChannels=64, int32_t levels=4, double dropout=0, bool showSizes=false) {
        this->levels=levels;
        this->showSizes=showSizes;

        for(int level=0; level<levels; level++)
        {
            contracting.push_back(nn::Sequential(
                                           nn::Conv2d(nn::Conv2dOptions(level==0?inChannels:featureChannels*(1<<(level-1)), featureChannels*(1<<level), 3).padding(1)),
                                           nn::ReLU(),
                                           nn::Dropout2d(dropout),
                                           nn::Conv2d(nn::Conv2dOptions(featureChannels*(1<<level), featureChannels*(1<<level), 3).padding(1)),
                                           nn::ReLU()
                                       ));
            register_module("contractingBlock"+std::to_string(level),contracting.back());

            downsampling.push_back(nn::MaxPool2d(nn::MaxPool2dOptions(2).stride(2)));
            register_module("downsampling"+std::to_string(level),downsampling.back());
        }

        bottleneck=nn::Sequential(
                        nn::Conv2d(nn::Conv2dOptions(featureChannels*(1<<(levels-1)), featureChannels*(1<<levels), 3).padding(1)),
                        nn::ReLU(),
                        nn::Dropout2d(dropout),
                        nn::Conv2d(nn::Conv2dOptions(featureChannels*(1<<levels), featureChannels*(1<<levels), 3).padding(1)),
                        nn::ReLU()
                    );
        register_module("bottleneck",bottleneck);

        for(int level=levels-1; level>=0; level--)
        {
            upsampling.push_front(nn::ConvTranspose2d(nn::ConvTranspose2dOptions(featureChannels*(1<<(level+1)), featureChannels*(1<<level), 3).stride(2).padding(1).output_padding(1)));
            register_module("upsampling"+std::to_string(level),upsampling.front());

            expanding.push_front(nn::Sequential(
                                       nn::Conv2d(nn::Conv2dOptions(featureChannels*(1<<(level+1)), featureChannels*(1<<level), 3).padding(1)),
                                       nn::ReLU(),
                                       nn::Dropout2d(dropout),
                                       nn::Conv2d(nn::Conv2dOptions(featureChannels*(1<<level), featureChannels*(1<<level), 3).padding(1)),
                                       nn::ReLU()
                                   ));
            register_module("expandingBlock"+std::to_string(level),expanding.front());
        }

        output=nn::Conv2d(nn::Conv2dOptions(featureChannels, outChannels, 1));
        register_module("output",output);
    }

    torch::Tensor forward(const torch::Tensor& inputTensor) {
        std::vector<Tensor> contractingTensor(levels);
        std::vector<Tensor> downsamplingTensor(levels);
        Tensor bottleneckTensor;
        std::vector<Tensor> upsamplingTensor(levels);
        std::vector<Tensor> expandingTensor(levels);
        Tensor outputTensor;

        for(int level=0; level<levels; level++)
        {
            contractingTensor[level]=contracting[level]->forward(level==0?inputTensor:downsamplingTensor[level-1]);
            downsamplingTensor[level]=downsampling[level]->forward(contractingTensor[level]);
        }

        bottleneckTensor=bottleneck->forward(downsamplingTensor.back());

        for(int level=levels-1; level>=0; level--)
        {
            upsamplingTensor[level]=upsampling[level]->forward(level==levels-1?bottleneckTensor:expandingTensor[level+1]);
            expandingTensor[level]=expanding[level]->forward(cat({contractingTensor[level],upsamplingTensor[level]},1));
        }

        outputTensor=output->forward(expandingTensor.front());

        if(showSizes)
        {
            std::cout << "input:  " << inputTensor.sizes() << endl;
            for(int level=0; level<levels; level++)
            {
                for(int i=0; i<level; i++) std::cout << " "; std::cout << " contracting" << level << ":  " << contractingTensor[level].sizes() << endl;
                for(int i=0; i<level; i++) std::cout << " "; std::cout << " downsampling" << level << ": " << downsamplingTensor[level].sizes() << endl;
            }
            for(int i=0; i<levels; i++) std::cout << " "; std::cout << " bottleneck:    " << bottleneckTensor.sizes() << endl;
            for(int level=levels-1; level>=0; level--)
            {
                for(int i=0; i<level; i++) std::cout << " "; std::cout << " upsampling" << level << ":  " << upsamplingTensor[level].sizes() << endl;
                for(int i=0; i<level; i++) std::cout << " "; std::cout << " expanding" << level << ":   " << expandingTensor[level].sizes() << endl;
            }
            std::cout << "output: " << outputTensor.sizes() << endl;
            showSizes=false;
        }

        return outputTensor;
    }

    //2d size you pass to the model must be a multiple of this
    int sizeMultiple() {return 1<<levels;}

    int levels;
    bool showSizes;

    std::deque<nn::Sequential> contracting;
    std::deque<nn::MaxPool2d> downsampling;
    nn::Sequential bottleneck;
    std::deque<nn::ConvTranspose2d> upsampling;
    std::deque<nn::Sequential> expanding;
    nn::Conv2d output{nullptr};
};
TORCH_MODULE_IMPL(UNet, UNetImpl);

#endif // UNET_H
