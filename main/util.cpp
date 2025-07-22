#include "util.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

Config Config::load(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + config_path);
    }
    
    nlohmann::json j;
    file >> j;
    
    Config config;
    
    config.data.train_path = j["data"]["train_path"];
    config.data.val_path = j["data"]["val_path"];
    config.data.charset = j["data"]["charset"];
    config.data.image_height = j["data"]["image_height"];
    config.data.image_width = j["data"]["image_width"];
    config.data.max_sequence_length = j["data"]["max_sequence_length"];
    config.data.vocab_size = j["data"]["vocab_size"];
    
    config.model.stn_channels = j["model"]["stn_channels"];
    config.model.stn_kernels = j["model"]["stn_kernels"];
    config.model.cnn_channels = j["model"]["cnn_channels"];
    config.model.lstm_hidden = j["model"]["lstm_hidden"];
    config.model.lstm_layers = j["model"]["lstm_layers"];
    config.model.attention_heads = j["model"]["attention_heads"];
    config.model.attention_dim = j["model"]["attention_dim"];
    config.model.dropout = j["model"]["dropout"];
    
    config.training.batch_size = j["training"]["batch_size"];
    config.training.gradient_accumulation_steps = j["training"]["gradient_accumulation_steps"];
    config.training.learning_rate = j["training"]["learning_rate"];
    config.training.weight_decay = j["training"]["weight_decay"];
    config.training.epochs = j["training"]["epochs"];
    config.training.warmup_steps = j["training"]["warmup_steps"];
    config.training.gradient_clip = j["training"]["gradient_clip"];
    config.training.mixed_precision = j["training"]["mixed_precision"];
    config.training.gradient_checkpointing = j["training"]["gradient_checkpointing"];
    
    config.augmentation.rotation = j["augmentation"]["rotation"];
    config.augmentation.perspective = j["augmentation"]["perspective"];
    config.augmentation.gaussian_noise = j["augmentation"]["gaussian_noise"];
    config.augmentation.blur_kernel = j["augmentation"]["blur_kernel"];
    config.augmentation.brightness_contrast = j["augmentation"]["brightness_contrast"];
    
    config.paths.model_save = j["paths"]["model_save"];
    config.paths.onnx_save = j["paths"]["onnx_save"];
    config.paths.log_dir = j["paths"]["log_dir"];
    
    return config;
}

CTCDecoder::CTCDecoder(const std::string& charset) : charset(charset), blank_index(charset.length()) {}

std::string CTCDecoder::decode_greedy(const torch::Tensor& log_probs) {
    auto probs = log_probs.exp();
    auto best_path = std::get<1>(torch::max(probs, 1));
    
    std::string result;
    int prev_idx = -1;
    
    for (int i = 0; i < best_path.size(0); i++) {
        int idx = best_path[i].item<int>();
        if (idx != blank_index && idx != prev_idx) {
            if (idx >= 0 && idx < charset.length()) {
                result += charset[idx];
            }
        }
        prev_idx = idx;
    }
    
    return result;
}

std::vector<std::string> CTCDecoder::decode_batch(const torch::Tensor& log_probs) {
    std::vector<std::string> results;
    int batch_size = log_probs.size(0);
    
    for (int b = 0; b < batch_size; b++) {
        results.push_back(decode_greedy(log_probs[b]));
    }
    
    return results;
}

double Metrics::character_error_rate(const std::string& pred, const std::string& target) {
    if (target.empty()) return pred.empty() ? 0.0 : 1.0;
    
    std::vector<std::vector<int>> dp(pred.length() + 1, std::vector<int>(target.length() + 1));
    
    for (int i = 0; i <= pred.length(); i++) dp[i][0] = i;
    for (int j = 0; j <= target.length(); j++) dp[0][j] = j;
    
    for (int i = 1; i <= pred.length(); i++) {
        for (int j = 1; j <= target.length(); j++) {
            if (pred[i-1] == target[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = 1 + std::min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]});
            }
        }
    }
    
    return static_cast<double>(dp[pred.length()][target.length()]) / target.length();
}

double Metrics::sequence_accuracy(const std::vector<std::string>& preds, 
                                 const std::vector<std::string>& targets) {
    if (preds.size() != targets.size()) return 0.0;
    
    int correct = 0;
    for (size_t i = 0; i < preds.size(); i++) {
        if (preds[i] == targets[i]) correct++;
    }
    
    return static_cast<double>(correct) / preds.size();
}

double Metrics::word_accuracy(const std::vector<std::string>& preds, 
                             const std::vector<std::string>& targets) {
    return sequence_accuracy(preds, targets);
}

Logger::Logger(const std::string& log_dir) : log_dir(log_dir) {
    std::filesystem::create_directories(log_dir);
    log_file.open(log_dir + "/training.log", std::ios::app);
}

void Logger::log_training(int epoch, double loss, double accuracy) {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "] ";
    ss << "Epoch " << epoch << " - Train Loss: " << std::fixed << std::setprecision(6) << loss;
    ss << ", Accuracy: " << std::fixed << std::setprecision(4) << accuracy;
    
    std::cout << ss.str() << std::endl;
    log_file << ss.str() << std::endl;
    log_file.flush();
}

void Logger::log_validation(int epoch, double loss, double cer, double seq_acc) {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "] ";
    ss << "Epoch " << epoch << " - Val Loss: " << std::fixed << std::setprecision(6) << loss;
    ss << ", CER: " << std::fixed << std::setprecision(4) << cer;
    ss << ", Seq Acc: " << std::fixed << std::setprecision(4) << seq_acc;
    
    std::cout << ss.str() << std::endl;
    log_file << ss.str() << std::endl;
    log_file.flush();
}

void Logger::log_info(const std::string& message) {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "] " << message;
    
    std::cout << ss.str() << std::endl;
    log_file << ss.str() << std::endl;
    log_file.flush();
}