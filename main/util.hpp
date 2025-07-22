#pragma once

#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

struct Config {
    struct Data {
        std::string train_path, val_path, charset;
        int image_height, image_width, max_sequence_length, vocab_size;
    } data;
    
    struct Model {
        std::vector<int> stn_channels, stn_kernels, cnn_channels, lstm_hidden, lstm_layers;
        int attention_heads, attention_dim;
        double dropout;
    } model;
    
    struct Training {
        int batch_size, gradient_accumulation_steps, epochs, warmup_steps;
        double learning_rate, weight_decay, gradient_clip;
        bool mixed_precision, gradient_checkpointing;
    } training;
    
    struct Augmentation {
        double rotation, perspective, gaussian_noise, brightness_contrast;
        int blur_kernel;
    } augmentation;
    
    struct Paths {
        std::string model_save, onnx_save, log_dir;
    } paths;
    
    static Config load(const std::string& config_path);
};

class CTCDecoder {
    std::string charset;
    int blank_index;
    
public:
    CTCDecoder(const std::string& charset);
    std::string decode_greedy(const torch::Tensor& log_probs);
    std::vector<std::string> decode_batch(const torch::Tensor& log_probs);
};

class Metrics {
public:
    static double character_error_rate(const std::string& pred, const std::string& target);
    static double sequence_accuracy(const std::vector<std::string>& preds, 
                                   const std::vector<std::string>& targets);
    static double word_accuracy(const std::vector<std::string>& preds, 
                               const std::vector<std::string>& targets);
};

class Logger {
    std::string log_dir;
    std::ofstream log_file;
    
public:
    Logger(const std::string& log_dir);
    void log_training(int epoch, double loss, double accuracy);
    void log_validation(int epoch, double loss, double cer, double seq_acc);
    void log_info(const std::string& message);
};