#include "model.hpp"
#include <torch/torch.h>

STNLocalizationNet::STNLocalizationNet() {
    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7).stride(1).padding(3)));
    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 5).stride(1).padding(2)));
    conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)));
    conv4 = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1)));
    
    fc1 = register_module("fc1", torch::nn::Linear(8192, 1024));
    fc2 = register_module("fc2", torch::nn::Linear(1024, 512));
    fc3 = register_module("fc3", torch::nn::Linear(512, 6));
    
    dropout1 = register_module("dropout1", torch::nn::Dropout(0.5));
    dropout2 = register_module("dropout2", torch::nn::Dropout(0.3));
    
    torch::NoGradGuard no_grad;
    fc3->weight.fill_(0);
    fc3->bias.copy_(torch::tensor({1, 0, 0, 0, 1, 0}, torch::kFloat));
}

torch::Tensor STNLocalizationNet::forward(torch::Tensor x) {
    x = torch::relu(conv1(x));
    x = torch::max_pool2d(x, 2, 2);
    
    x = torch::relu(conv2(x));
    x = torch::max_pool2d(x, 2, 2);
    
    x = torch::relu(conv3(x));
    x = torch::max_pool2d(x, 2, 2);
    
    x = torch::relu(conv4(x));
    x = torch::max_pool2d(x, 2, 2);
    
    x = x.view({x.size(0), -1});
    
    x = torch::relu(fc1(x));
    x = dropout1(x);
    
    x = torch::relu(fc2(x));
    x = dropout2(x);
    
    x = fc3(x);
    
    return x;
}

ResidualBlock::ResidualBlock(int in_channels, int out_channels, int stride) 
    : use_shortcut(stride != 1 || in_channels != out_channels) {
    
    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(stride).padding(1).bias(false)));
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channels));
    
    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3).stride(1).padding(1).bias(false)));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));
    
    if (use_shortcut) {
        shortcut = register_module("shortcut", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(stride).bias(false)));
        bn_shortcut = register_module("bn_shortcut", torch::nn::BatchNorm2d(out_channels));
    }
}

torch::Tensor ResidualBlock::forward(torch::Tensor x) {
    auto identity = x;
    
    auto out = torch::relu(bn1(conv1(x)));
    out = bn2(conv2(out));
    
    if (use_shortcut) {
        identity = bn_shortcut(shortcut(x));
    }
    
    out += identity;
    return torch::relu(out);
}

CNNBackbone::CNNBackbone() {
    stem = register_module("stem", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3).bias(false)));
    stem_bn = register_module("stem_bn", torch::nn::BatchNorm2d(64));
    
    block1 = register_module("block1", torch::nn::Sequential(
        ResidualBlock(64, 64),
        ResidualBlock(64, 64),
        ResidualBlock(64, 64)
    ));
    
    block2 = register_module("block2", torch::nn::Sequential(
        ResidualBlock(64, 128, 2),
        ResidualBlock(128, 128),
        ResidualBlock(128, 128),
        ResidualBlock(128, 128)
    ));
    
    block3 = register_module("block3", torch::nn::Sequential(
        ResidualBlock(128, 256, 2),
        ResidualBlock(256, 256),
        ResidualBlock(256, 256),
        ResidualBlock(256, 256),
        ResidualBlock(256, 256),
        ResidualBlock(256, 256)
    ));
    
    block4 = register_module("block4", torch::nn::Sequential(
        ResidualBlock(256, 512, 2),
        ResidualBlock(512, 512),
        ResidualBlock(512, 512)
    ));
}

torch::Tensor CNNBackbone::forward(torch::Tensor x) {
    x = torch::relu(stem_bn(stem(x)));
    x = torch::max_pool2d(x, 3, 2, 1);
    
    x = block1->forward(x);
    x = block2->forward(x);
    x = block3->forward(x);
    x = block4->forward(x);
    
    return x;
}

MultiHeadAttention::MultiHeadAttention(int d_model, int n_heads, double dropout_rate) 
    : n_heads(n_heads), d_model(d_model), d_k(d_model / n_heads) {
    
    q_linear = register_module("q_linear", torch::nn::Linear(d_model, d_model));
    k_linear = register_module("k_linear", torch::nn::Linear(d_model, d_model));
    v_linear = register_module("v_linear", torch::nn::Linear(d_model, d_model));
    out = register_module("out", torch::nn::Linear(d_model, d_model));
    dropout = register_module("dropout", torch::nn::Dropout(dropout_rate));
}

torch::Tensor MultiHeadAttention::forward(torch::Tensor x) {
    int batch_size = x.size(0);
    int seq_len = x.size(1);
    
    auto q = q_linear(x).view({batch_size, seq_len, n_heads, d_k}).transpose(1, 2);
    auto k = k_linear(x).view({batch_size, seq_len, n_heads, d_k}).transpose(1, 2);
    auto v = v_linear(x).view({batch_size, seq_len, n_heads, d_k}).transpose(1, 2);
    
    auto scores = torch::matmul(q, k.transpose(-2, -1)) / sqrt(d_k);
    auto attn = torch::softmax(scores, -1);
    attn = dropout(attn);
    
    auto context = torch::matmul(attn, v);
    context = context.transpose(1, 2).contiguous().view({batch_size, seq_len, d_model});
    
    return out(context);
}

CRNN::CRNN(int vocab_size) : vocab_size(vocab_size) {
    stn = register_module("stn", std::make_shared<STNLocalizationNet>());
    cnn = register_module("cnn", std::make_shared<CNNBackbone>());
    
    auto lstm_options1 = torch::nn::LSTMOptions(512, 256).num_layers(2).dropout(0.3).bidirectional(true).batch_first(true);
    lstm1 = register_module("lstm1", torch::nn::LSTM(lstm_options1));
    
    auto lstm_options2 = torch::nn::LSTMOptions(512, 256).num_layers(2).dropout(0.3).bidirectional(true).batch_first(true);
    lstm2 = register_module("lstm2", torch::nn::LSTM(lstm_options2));
    
    auto lstm_options3 = torch::nn::LSTMOptions(512, 128).num_layers(1).bidirectional(true).batch_first(true);
    lstm3 = register_module("lstm3", torch::nn::LSTM(lstm_options3));
    
    lstm_dropout = register_module("lstm_dropout", torch::nn::Dropout(0.3));
    attention = register_module("attention", std::make_shared<MultiHeadAttention>(256, 8));
    classifier = register_module("classifier", torch::nn::Linear(256, vocab_size));
}

torch::Tensor CRNN::apply_stn(torch::Tensor x, torch::Tensor theta) {
    auto grid = torch::nn::functional::affine_grid(theta.view({-1, 2, 3}), x.sizes());
    return torch::nn::functional::grid_sample(x, grid);
}

torch::Tensor CRNN::forward(torch::Tensor x) {
    auto theta = stn->forward(x);
    x = apply_stn(x, theta);
    
    x = cnn->forward(x);
    
    auto batch_size = x.size(0);
    auto channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);
    
    x = x.permute({0, 3, 1, 2});
    x = x.contiguous().view({batch_size, width, channels * height});
    
    auto lstm_out1 = std::get<0>(lstm1->forward(x));
    lstm_out1 = lstm_dropout(lstm_out1);
    
    auto lstm_out2 = std::get<0>(lstm2->forward(lstm_out1));
    lstm_out2 = lstm_dropout(lstm_out2);
    
    auto lstm_out3 = std::get<0>(lstm3->forward(lstm_out2));
    
    auto attn_out = attention->forward(lstm_out3);
    auto output = classifier(attn_out);
    
    return output.log_softmax(2);
}