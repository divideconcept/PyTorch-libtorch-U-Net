#ifndef PARTIALCONV_H
#define PARTIALCONV_H

// Partial Convolution for PyTorch C++
// -------------------------------------------------------
// Robin Lobel, March 2020 - requires libtorch 1.4 and higher - Qt compatible
// https://github.com/divideconcept/PyTorch-libtorch-U-Net
//
// Partial Convolution fixes the errors introduced by zero-padding convolution, while keeping the convenience of padding
// Partial Mask Convolution supports additionally a mask

// Adapted from https://github.com/NVIDIA/partialconv
// NVIDIA Paper: https://arxiv.org/pdf/1811.11718.pdf

//Qt compatibility
#ifdef QT_CORE_LIB
    #undef slots
#endif
#include<torch/torch.h>
#ifdef QT_CORE_LIB
    #define slots Q_SLOTS
#endif

class PartialMaskConv2dImpl : public torch::nn::Conv2dImpl {
public:
    PartialMaskConv2dImpl(torch::nn::Conv2dOptions options_, bool multi_channel=false) : torch::nn::Conv2dImpl(options_)
    {
        // whether the mask is multi-channel or not
        this->multi_channel = multi_channel;

        if(multi_channel)
            weight_maskUpdater = torch::ones({options.out_channels(), options.in_channels(), options.kernel_size()->at(0), options.kernel_size()->at(1)});
        else
            weight_maskUpdater = torch::ones({1, 1, options.kernel_size()->at(0), options.kernel_size()->at(1)});


        slide_winsize = weight_maskUpdater.size(1) * weight_maskUpdater.size(2) * weight_maskUpdater.size(3);

        last_size={0,0,0,0};

        update_mask = torch::Tensor();
        mask_ratio = torch::Tensor();
    }

    std::tuple<torch::Tensor,torch::Tensor> forward(std::tuple<torch::Tensor,torch::Tensor> inputs) {
        torch::Tensor input=std::get<0>(inputs); torch::Tensor mask_in=std::get<1>(inputs);
        torch::Tensor mask;
        if(mask_in.defined() || last_size != input.sizes())
        {
            last_size=input.sizes();
            {
                torch::NoGradGuard no_grad;
                if(weight_maskUpdater.scalar_type()!=input.scalar_type() || weight_maskUpdater.device()!=input.device())
                    weight_maskUpdater = weight_maskUpdater.to(input);

                if(!mask_in.defined())
                {
                    // if mask is not provided, create a mask
                    if(multi_channel)
                        mask = torch::ones({input.data().size(0), input.data().size(1), input.data().size(2), input.data().size(3)}).to(input);
                    else
                        mask = torch::ones({1, 1, input.data().size(2), input.data().size(3)}).to(input);
                } else {
                    mask = mask_in;
                }

                update_mask = torch::nn::functional::conv2d(mask, weight_maskUpdater, torch::nn::functional::Conv2dFuncOptions().bias(torch::Tensor()).stride(options.stride()).padding(options.padding()).dilation(options.dilation()).groups(1));

                if(!mask_in.defined())
                    mask_ratio = slide_winsize/update_mask;
                else {
                    mask_ratio = slide_winsize/(update_mask + (mask.type().scalarType()==torch::kFloat?1e-8:1e-6));
                    update_mask = torch::clamp(update_mask, 0, 1);
                    mask_ratio = torch::mul(mask_ratio, update_mask);
                }
            }
        }

        torch::Tensor raw_out = Conv2dImpl::forward(mask_in.defined()?torch::mul(input, mask):input);

        torch::Tensor output;
        if(this->bias.defined())
        {
            torch::Tensor bias_view = this->bias.view({1, options.out_channels(), 1, 1});
            output = torch::mul(raw_out - bias_view, mask_ratio) + bias_view;
            if(mask_in.defined()) output = torch::mul(output, update_mask);
        } else
            output = torch::mul(raw_out, mask_ratio);

        return std::make_tuple(output, update_mask);
    }

private:
    bool multi_channel;
    torch::Tensor weight_maskUpdater;
    float slide_winsize;
    torch::IntArrayRef last_size;
    torch::Tensor update_mask;
    torch::Tensor mask_ratio;
};
TORCH_MODULE(PartialMaskConv2d);

class PartialConv2dImpl : public PartialMaskConv2dImpl {
public:
    PartialConv2dImpl(torch::nn::Conv2dOptions options_) : PartialMaskConv2dImpl(options_,false) { }
    torch::Tensor forward(torch::Tensor input) { return std::get<0>(PartialMaskConv2dImpl::forward(std::make_tuple(input,torch::Tensor()))); }
};
TORCH_MODULE(PartialConv2d);


class PartialMaskConv1dImpl : public torch::nn::Conv1dImpl {
public:
    PartialMaskConv1dImpl(torch::nn::Conv1dOptions options_, bool multi_channel=false) : torch::nn::Conv1dImpl(options_)
    {
        // whether the mask is multi-channel or not
        this->multi_channel = multi_channel;

        if(multi_channel)
            weight_maskUpdater = torch::ones({options.out_channels(), options.in_channels(), options.kernel_size()->at(0)});
        else
            weight_maskUpdater = torch::ones({1, 1, options.kernel_size()->at(0)});


        slide_winsize = weight_maskUpdater.size(1) * weight_maskUpdater.size(2);

        last_size={0,0,0};

        update_mask = torch::Tensor();
        mask_ratio = torch::Tensor();
    }

    std::tuple<torch::Tensor,torch::Tensor> forward(std::tuple<torch::Tensor,torch::Tensor> inputs) {
        torch::Tensor input=std::get<0>(inputs); torch::Tensor mask_in=std::get<1>(inputs);
        torch::Tensor mask;
        if(mask_in.defined() || last_size != input.sizes())
        {
            last_size=input.sizes();
            {
                torch::NoGradGuard no_grad;
                if(weight_maskUpdater.scalar_type()!=input.scalar_type() || weight_maskUpdater.device()!=input.device())
                    weight_maskUpdater = weight_maskUpdater.to(input);

                if(!mask_in.defined())
                {
                    // if mask is not provided, create a mask
                    if(multi_channel)
                        mask = torch::ones({input.data().size(0), input.data().size(1), input.data().size(2)}).to(input);
                    else
                        mask = torch::ones({1, 1, input.data().size(2)}).to(input);
                } else {
                    mask = mask_in;
                }

                update_mask = torch::nn::functional::conv1d(mask, weight_maskUpdater, torch::nn::functional::Conv1dFuncOptions().bias(torch::Tensor()).stride(options.stride()).padding(options.padding()).dilation(options.dilation()).groups(1));

                if(!mask_in.defined())
                    mask_ratio = slide_winsize/update_mask;
                else {
                    mask_ratio = slide_winsize/(update_mask + (mask.type().scalarType()==torch::kFloat?1e-8:1e-6));
                    update_mask = torch::clamp(update_mask, 0, 1);
                    mask_ratio = torch::mul(mask_ratio, update_mask);
                }
            }
        }

        torch::Tensor raw_out = Conv1dImpl::forward(mask_in.defined()?torch::mul(input, mask):input);

        torch::Tensor output;
        if(this->bias.defined())
        {
            torch::Tensor bias_view = this->bias.view({1, options.out_channels(), 1});
            output = torch::mul(raw_out - bias_view, mask_ratio) + bias_view;
            if(mask_in.defined()) output = torch::mul(output, update_mask);
        } else
            output = torch::mul(raw_out, mask_ratio);

        return std::make_tuple(output, update_mask);
    }

private:
    bool multi_channel;
    torch::Tensor weight_maskUpdater;
    float slide_winsize;
    torch::IntArrayRef last_size;
    torch::Tensor update_mask;
    torch::Tensor mask_ratio;
};
TORCH_MODULE(PartialMaskConv1d);

class PartialConv1dImpl : public PartialMaskConv1dImpl {
public:
    PartialConv1dImpl(torch::nn::Conv1dOptions options_) : PartialMaskConv1dImpl(options_,false) { }
    torch::Tensor forward(torch::Tensor input) { return std::get<0>(PartialMaskConv1dImpl::forward(std::make_tuple(input,torch::Tensor()))); }
};
TORCH_MODULE(PartialConv1d);

#endif // PARTIALCONV_H
