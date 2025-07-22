#include <torch/torch.h>
#include <cxxopts.hpp>
#include <iostream>
#include <memory>
#include <chrono>

#include "model.hpp"
#include "data.hpp"
#include "util.hpp"

void train_model(const Config& config) {
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Using device: " << device << std::endl;
    
    auto model = std::make_shared<CRNN>(config.data.vocab_size);
    model->to(device);
    
    auto train_dataset = CaptchaDataset(config.data.train_path, config.data.charset, 
                                       config.data.image_height, config.data.image_width, true)
                        .map(torch::data::transforms::Stack<>());
    
    auto val_dataset = CaptchaDataset(config.data.val_path, config.data.charset,
                                     config.data.image_height, config.data.image_width, false)
                      .map(torch::data::transforms::Stack<>());
    
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), config.training.batch_size);
    
    auto val_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(val_dataset), config.training.batch_size);
    
    torch::optim::AdamW optimizer(model->parameters(), 
                                 torch::optim::AdamWOptions(config.training.learning_rate)
                                 .weight_decay(config.training.weight_decay));
    
    auto scheduler = torch::optim::CosineAnnealingLRScheduler(optimizer, 15);
    
    auto ctc_loss = torch::nn::CTCLoss(torch::nn::CTCLossOptions().blank(config.data.vocab_size - 1));
    
    CTCDecoder decoder(config.data.charset);
    Logger logger(config.paths.log_dir);
    
    torch::cuda::setCurrentStream(torch::cuda::getDefaultStream());
    torch::GradScaler scaler;
    
    double best_val_acc = 0.0;
    int step = 0;
    
    logger.log_info("Starting training...");
    
    for (int epoch = 0; epoch < config.training.epochs; epoch++) {
        model->train();
        double epoch_loss = 0.0;
        int num_batches = 0;
        int correct_predictions = 0;
        int total_predictions = 0;
        
        for (auto& batch : *train_loader) {
            auto data = batch.data.to(device);
            auto targets = batch.target;
            
            std::vector<torch::Tensor> target_tensors;
            std::vector<int64_t> target_lengths;
            
            for (int i = 0; i < targets.size(0); i++) {
                auto target = targets[i];
                target_tensors.push_back(target.to(device));
                target_lengths.push_back(target.size(0));
            }
            
            auto target_concat = torch::cat(target_tensors, 0);
            auto target_lengths_tensor = torch::tensor(target_lengths, torch::kLong).to(device);
            
            if (config.training.mixed_precision) {
                torch::autocast::autocast_mode autocast_guard(torch::kCUDA, true);
                
                auto outputs = model->forward(data);
                auto input_lengths = torch::full({outputs.size(0)}, outputs.size(1), torch::kLong).to(device);
                
                auto loss = ctc_loss->forward(outputs.transpose(0, 1), target_concat, 
                                            input_lengths, target_lengths_tensor);
                
                loss = loss / config.training.gradient_accumulation_steps;
                scaler.scale(loss).backward();
                
                epoch_loss += loss.item<double>() * config.training.gradient_accumulation_steps;
            } else {
                auto outputs = model->forward(data);
                auto input_lengths = torch::full({outputs.size(0)}, outputs.size(1), torch::kLong).to(device);
                
                auto loss = ctc_loss->forward(outputs.transpose(0, 1), target_concat, 
                                            input_lengths, target_lengths_tensor);
                
                loss = loss / config.training.gradient_accumulation_steps;
                loss.backward();
                
                epoch_loss += loss.item<double>() * config.training.gradient_accumulation_steps;
            }
            
            if ((step + 1) % config.training.gradient_accumulation_steps == 0) {
                if (config.training.mixed_precision) {
                    scaler.unscale_(optimizer);
                    torch::nn::utils::clip_grad_norm_(model->parameters(), config.training.gradient_clip);
                    scaler.step(optimizer);
                    scaler.update();
                } else {
                    torch::nn::utils::clip_grad_norm_(model->parameters(), config.training.gradient_clip);
                    optimizer.step();
                }
                
                optimizer.zero_grad();
                
                if (step < config.training.warmup_steps) {
                    double warmup_lr = config.training.learning_rate * (step + 1) / config.training.warmup_steps;
                    for (auto& group : optimizer.param_groups()) {
                        group.options().set_lr(warmup_lr);
                    }
                }
            }
            
            step++;
            num_batches++;
            
            if (num_batches % 100 == 0) {
                std::cout << "Batch " << num_batches << "/" << train_loader->size().value() 
                         << " - Loss: " << epoch_loss / num_batches << std::endl;
            }
        }
        
        if (step >= config.training.warmup_steps) {
            scheduler.step();
        }
        
        epoch_loss /= num_batches;
        double train_accuracy = static_cast<double>(correct_predictions) / total_predictions;
        
        model->eval();
        double val_loss = 0.0;
        std::vector<std::string> all_predictions, all_targets;
        int val_batches = 0;
        
        {
            torch::NoGradGuard no_grad;
            for (auto& batch : *val_loader) {
                auto data = batch.data.to(device);
                auto targets = batch.target;
                
                std::vector<torch::Tensor> target_tensors;
                std::vector<int64_t> target_lengths;
                std::vector<std::string> batch_targets;
                
                for (int i = 0; i < targets.size(0); i++) {
                    auto target = targets[i];
                    target_tensors.push_back(target.to(device));
                    target_lengths.push_back(target.size(0));
                    
                    std::string target_str;
                    for (int j = 0; j < target.size(0); j++) {
                        int idx = target[j].item<int>();
                        if (idx >= 0 && idx < config.data.charset.length()) {
                            target_str += config.data.charset[idx];
                        }
                    }
                    batch_targets.push_back(target_str);
                }
                
                auto target_concat = torch::cat(target_tensors, 0);
                auto target_lengths_tensor = torch::tensor(target_lengths, torch::kLong).to(device);
                
                auto outputs = model->forward(data);
                auto input_lengths = torch::full({outputs.size(0)}, outputs.size(1), torch::kLong).to(device);
                
                auto loss = ctc_loss->forward(outputs.transpose(0, 1), target_concat, 
                                            input_lengths, target_lengths_tensor);
                val_loss += loss.item<double>();
                
                auto predictions = decoder.decode_batch(outputs.cpu());
                all_predictions.insert(all_predictions.end(), predictions.begin(), predictions.end());
                all_targets.insert(all_targets.end(), batch_targets.begin(), batch_targets.end());
                
                val_batches++;
            }
        }
        
        val_loss /= val_batches;
        double avg_cer = 0.0;
        for (size_t i = 0; i < all_predictions.size(); i++) {
            avg_cer += Metrics::character_error_rate(all_predictions[i], all_targets[i]);
        }
        avg_cer /= all_predictions.size();
        
        double seq_accuracy = Metrics::sequence_accuracy(all_predictions, all_targets);
        
        logger.log_training(epoch + 1, epoch_loss, train_accuracy);
        logger.log_validation(epoch + 1, val_loss, avg_cer, seq_accuracy);
        
        if (seq_accuracy > best_val_acc) {
            best_val_acc = seq_accuracy;
            torch::save(model, config.paths.model_save);
            logger.log_info("New best model saved with accuracy: " + std::to_string(seq_accuracy));
        }
        
        if ((epoch + 1) % 10 == 0) {
            std::string checkpoint_path = config.paths.log_dir + "/checkpoint_epoch_" + std::to_string(epoch + 1) + ".pt";
            torch::save(model, checkpoint_path);
        }
    }
    
    logger.log_info("Training completed!");
}

void inference_mode(const Config& config, const std::string& image_path) {
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    
    auto model = std::make_shared<CRNN>(config.data.vocab_size);
    torch::load(model, config.paths.model_save);
    model->to(device);
    model->eval();
    
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return;
    }
    
    cv::resize(image, image, cv::Size(config.data.image_width, config.data.image_height));
    auto tensor_image = mat_to_tensor(image).unsqueeze(0).to(device);
    
    CTCDecoder decoder(config.data.charset);
    
    torch::NoGradGuard no_grad;
    auto output = model->forward(tensor_image);
    auto prediction = decoder.decode_greedy(output[0].cpu());
    
    std::cout << "Prediction: " << prediction << std::endl;
}

int main(int argc, char** argv) {
    cxxopts::Options options("captcha_solver", "CAPTCHA Solver with STN+CRNN+CTC");
    
    options.add_options()
        ("t,train", "Training mode")
        ("i,infer", "Inference mode", cxxopts::value<std::string>())
        ("c,config", "Config file path", cxxopts::value<std::string>()->default_value("configs.json"))
        ("h,help", "Print usage");
    
    auto result = options.parse(argc, argv);
    
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }
    
    std::string config_path = result["config"].as<std::string>();
    Config config = Config::load(config_path);
    
    if (result.count("train")) {
        train_model(config);
    } else if (result.count("infer")) {
        std::string image_path = result["infer"].as<std::string>();
        inference_mode(config, image_path);
    } else {
        std::cout << "Please specify --train or --infer <image_path>" << std::endl;
        std::cout << options.help() << std::endl;
        return 1;
    }
    
    return 0;
}