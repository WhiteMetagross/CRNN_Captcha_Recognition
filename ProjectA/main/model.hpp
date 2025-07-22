#pragma once

#include <torch/torch.h>
#include <memory>
#include <vector>

struct STNLocalizationNet : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    torch::nn::Dropout dropout1{nullptr}, dropout2{nullptr};
    
    STNLocalizationNet();
    torch::Tensor forward(torch::Tensor x);
    void register_modules();
};

struct ResidualBlock : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, shortcut{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn_shortcut{nullptr};
    bool use_shortcut;
    
    ResidualBlock(int in_channels, int out_channels, int stride = 1);
    torch::Tensor forward(torch::Tensor x);
    void register_modules();
};

struct CNNBackbone : torch::nn::Module {
    torch::nn::Conv2d stem{nullptr};
    torch::nn::BatchNorm2d stem_bn{nullptr};
    torch::nn::Sequential block1{nullptr}, block2{nullptr}, block3{nullptr}, block4{nullptr};
    
    CNNBackbone();
    torch::Tensor forward(torch::Tensor x);
    void register_modules();
};

struct MultiHeadAttention : torch::nn::Module {
    torch::nn::Linear q_linear{nullptr}, k_linear{nullptr}, v_linear{nullptr}, out{nullptr};
    torch::nn::Dropout dropout{nullptr};
    int n_heads, d_model, d_k;
    
    MultiHeadAttention(int d_model, int n_heads, double dropout_rate = 0.1);
    torch::Tensor forward(torch::Tensor x);
    void register_modules();
};

struct CRNN : torch::nn::Module {
    STNLocalizationNet stn{nullptr};
    CNNBackbone cnn{nullptr};
    torch::nn::LSTM lstm1{nullptr}, lstm2{nullptr}, lstm3{nullptr};
    torch::nn::Dropout lstm_dropout{nullptr};
    MultiHeadAttention attention{nullptr};
    torch::nn::Linear classifier{nullptr};
    
    int vocab_size;
    
    CRNN(int vocab_size);
    torch::Tensor forward(torch::Tensor x);
    torch::Tensor apply_stn(torch::Tensor x, torch::Tensor theta);
    void register_modules();
};

torch::Tensor grid_sample(torch::Tensor input, torch::Tensor grid, bool align_corners = false);