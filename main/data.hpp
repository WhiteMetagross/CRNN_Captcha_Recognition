#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <filesystem>

struct CaptchaDataset : torch::data::Dataset<CaptchaDataset> {
    std::vector<std::string> image_paths;
    std::vector<std::string> labels;
    std::string charset;
    int image_height, image_width;
    bool is_training;
    double rotation_range, perspective_range, noise_std, blur_kernel, brightness_contrast;
    
    CaptchaDataset(const std::string& root_dir, const std::string& charset, 
                   int height, int width, bool training = true);
    
    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;
    
private:
    void load_data(const std::string& root_dir);
    torch::Tensor augment_image(const cv::Mat& image);
    torch::Tensor encode_label(const std::string& label);
    std::string extract_label_from_filename(const std::string& filename);
};

torch::Tensor mat_to_tensor(const cv::Mat& mat);
cv::Mat tensor_to_mat(const torch::Tensor& tensor);