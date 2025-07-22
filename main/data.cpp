#include "data.hpp"
#include <random>
#include <algorithm>
#include <sstream>

CaptchaDataset::CaptchaDataset(const std::string& root_dir, const std::string& charset,
                               int height, int width, bool training)
    : charset(charset), image_height(height), image_width(width), is_training(training),
      rotation_range(12.0), perspective_range(0.2), noise_std(0.05), 
      blur_kernel(3), brightness_contrast(0.25) {
    load_data(root_dir);
}

void CaptchaDataset::load_data(const std::string& root_dir) {
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root_dir)) {
        if (entry.is_regular_file()) {
            std::string path = entry.path().string();
            std::string ext = entry.path().extension().string();
            
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                std::string label = extract_label_from_filename(entry.path().filename().string());
                if (!label.empty()) {
                    image_paths.push_back(path);
                    labels.push_back(label);
                }
            }
        }
    }
}

std::string CaptchaDataset::extract_label_from_filename(const std::string& filename) {
    std::string label = filename;
    size_t dot_pos = label.find_last_of('.');
    if (dot_pos != std::string::npos) {
        label = label.substr(0, dot_pos);
    }
    
    std::string result;
    for (char c : label) {
        if (charset.find(c) != std::string::npos) {
            result += c;
        }
    }
    return result;
}

torch::data::Example<> CaptchaDataset::get(size_t index) {
    cv::Mat image = cv::imread(image_paths[index]);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + image_paths[index]);
    }
    
    cv::resize(image, image, cv::Size(image_width, image_height));
    
    torch::Tensor tensor_image;
    if (is_training) {
        tensor_image = augment_image(image);
    } else {
        tensor_image = mat_to_tensor(image);
    }
    
    torch::Tensor encoded_label = encode_label(labels[index]);
    
    return {tensor_image, encoded_label};
}

torch::optional<size_t> CaptchaDataset::size() const {
    return image_paths.size();
}

torch::Tensor CaptchaDataset::augment_image(const cv::Mat& image) {
    cv::Mat augmented = image.clone();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    
    if (is_training) {
        std::uniform_real_distribution<> rotation_dis(-rotation_range, rotation_range);
        double angle = rotation_dis(gen);
        cv::Point2f center(augmented.cols / 2.0, augmented.rows / 2.0);
        cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::warpAffine(augmented, augmented, rotation_matrix, augmented.size());
        
        if (dis(gen) > 0) {
            cv::Point2f src_points[4] = {
                {0, 0}, {(float)augmented.cols, 0}, 
                {(float)augmented.cols, (float)augmented.rows}, {0, (float)augmented.rows}
            };
            cv::Point2f dst_points[4];
            for (int i = 0; i < 4; i++) {
                dst_points[i] = src_points[i];
                dst_points[i].x += perspective_range * augmented.cols * dis(gen) * 0.5;
                dst_points[i].y += perspective_range * augmented.rows * dis(gen) * 0.5;
            }
            cv::Mat perspective_matrix = cv::getPerspectiveTransform(src_points, dst_points);
            cv::warpPerspective(augmented, augmented, perspective_matrix, augmented.size());
        }
        
        if (dis(gen) > 0) {
            cv::Mat noise = cv::Mat::zeros(augmented.size(), CV_8UC3);
            cv::randn(noise, 0, noise_std * 255);
            augmented += noise;
        }
        
        if (dis(gen) > 0) {
            int kernel_size = 1 + (std::abs((int)(dis(gen) * blur_kernel)) % blur_kernel);
            if (kernel_size % 2 == 0) kernel_size++;
            cv::GaussianBlur(augmented, augmented, cv::Size(kernel_size, kernel_size), 0);
        }
        
        double brightness = brightness_contrast * dis(gen);
        double contrast = 1.0 + brightness_contrast * dis(gen);
        augmented.convertTo(augmented, -1, contrast, brightness * 255);
    }
    
    return mat_to_tensor(augmented);
}

torch::Tensor CaptchaDataset::encode_label(const std::string& label) {
    std::vector<int64_t> encoded;
    for (char c : label) {
        auto pos = charset.find(c);
        if (pos != std::string::npos) {
            encoded.push_back(static_cast<int64_t>(pos));
        }
    }
    return torch::tensor(encoded, torch::kLong);
}

torch::Tensor mat_to_tensor(const cv::Mat& mat) {
    cv::Mat float_mat;
    mat.convertTo(float_mat, CV_32F, 1.0/255.0);
    
    auto tensor = torch::from_blob(float_mat.data, {1, mat.rows, mat.cols, mat.channels()}, torch::kFloat);
    tensor = tensor.permute({0, 3, 1, 2}).clone();
    return tensor.squeeze(0);
}

cv::Mat tensor_to_mat(const torch::Tensor& tensor) {
    auto cpu_tensor = tensor.to(torch::kCPU);
    if (cpu_tensor.dim() == 4) cpu_tensor = cpu_tensor.squeeze(0);
    
    cpu_tensor = cpu_tensor.permute({1, 2, 0}).contiguous();
    cpu_tensor = cpu_tensor * 255;
    cpu_tensor = cpu_tensor.clamp(0, 255).to(torch::kUInt8);
    
    cv::Mat mat(cpu_tensor.size(0), cpu_tensor.size(1), CV_8UC3, cpu_tensor.data_ptr<uint8_t>());
    return mat.clone();
}