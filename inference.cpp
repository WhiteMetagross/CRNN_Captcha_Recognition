//This program performs inference on an ONNX model for captcha recognition.
//It reads an image, preprocesses it, runs the model, and decodes the output using CTC decoding.
//It uses OpenCV for image processing and ONNX Runtime for model inference.

#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <onnxruntime_cxx_api.h>

//Function to decode the output of the model.
//This function implements a simple CTC (Connectionist Temporal Classification) decoding algorithm.
std::string ctc_decode(const float* data, const std::vector<int64_t>& shape) {
    const std::string charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    const int blank_idx = charset.length();
    
    std::string result = "";
    int last_idx = -1;

    int sequence_length = shape[1];
    int num_classes = shape[2];

    for (int t = 0; t < sequence_length; ++t) {
        const float* time_step_data = data + t * num_classes;
        int max_idx = std::distance(time_step_data, std::max_element(time_step_data, time_step_data + num_classes));

        if (max_idx != blank_idx && max_idx != last_idx) {
            result += charset[max_idx];
        }
        last_idx = max_idx;
    }
    return result;
}

//Main function to perform inference.
//It initializes the ONNX Runtime, reads the input image, preprocesses it, runs the model, and decodes the output.
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_onnx_model> <path_to_image>" << std::endl;
        return -1;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];

    const int IMG_HEIGHT = 64;
    const int IMG_WIDTH = 256;
    const std::vector<float> NORM_MEAN = {0.485f, 0.456f, 0.406f};
    const std::vector<float> NORM_STD = {0.229f, 0.224f, 0.225f};

    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Could not read image at " << image_path << std::endl;
        return -1;
    }

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(IMG_WIDTH, IMG_HEIGHT));

    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255.0);
    cv::subtract(resized_image, NORM_MEAN, resized_image);
    cv::divide(resized_image, NORM_STD, resized_image);

    cv::Mat nchw_image = cv::dnn::blobFromImage(resized_image);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "captcha_inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    Ort::Session session(env, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    const char* input_name = session.GetInputName(0, allocator);
    const char* output_name = session.GetOutputName(0, allocator);
    
    std::vector<int64_t> input_shape = {1, 3, IMG_HEIGHT, IMG_WIDTH};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)nchw_image.data, nchw_image.total(), input_shape.data(), input_shape.size());

    std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, &output_name, 1);

    const float* output_data = output_tensors[0].GetTensorData<float>();
    std::vector<int64_t> output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    
    std::string prediction = ctc_decode(output_data, output_shape);

    std::cout << "Prediction: " << prediction << std::endl;

    return 0;
}
