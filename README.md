Of course\! Here is a comprehensive `README.md` file for your project, including instructions on how to install dependencies, prepare the dataset, and train and run the model.

-----

# C++ CAPTCHA Solver with STN-CRNN

This project is a C++ implementation of a CAPTCHA recognition system using a deep learning model. The model architecture is a Convolutional Recurrent Neural Network (CRNN) with a Spatial Transformer Network (STN) at the input and a Connectionist Temporal Classification (CTC) loss function for training. The implementation uses LibTorch, the C++ frontend for PyTorch.

## Features

  * **Spatial Transformer Network (STN):** Helps in correcting the geometric distortions in the input CAPTCHA images.
  * **Convolutional Neural Network (CNN):** A ResNet-based backbone is used for feature extraction from the images.
  * **Recurrent Neural Network (RNN):** Multi-layered LSTMs with bidirectional processing to capture sequential information from the features.
  * **Multi-Head Attention:** An attention mechanism to focus on the most relevant features.
  * **CTC Loss:** Enables training the model without character-level annotations for the alignment.
  * **Data Augmentation:** Includes various augmentation techniques like rotation, perspective transformation, noise, and blur to improve model robustness.
  * **Training and Inference:** The project supports both training the model from scratch and running inference on new images.
  * **ONNX Export:** A Python script is provided to export the trained model to the ONNX format for cross-platform deployment.

## Project Structure

```
aster_crnn_ctc_stn/
├── CMakeLists.txt
├── README.md
├── configs.json
├── src/
│   ├── main.cpp
│   ├── model.hpp
│   ├── model.cpp
│   ├── data.hpp
│   ├── data.cpp
│   ├── util.hpp
│   └── util.cpp
├── scripts/
│   └── export_onnx.py
└── models/
```

## Prerequisites

Before you begin, ensure you have the following dependencies installed:

  * **CMake:** For building the project.
  * **A C++17 compliant compiler:** (e.g., GCC, Clang, MSVC).
  * **LibTorch:** The C++ distribution of PyTorch.
  * **OpenCV:** For image processing and data loading.
  * **ONNX Runtime:** Required for linking against ONNX, as specified in the CMake file.

### Installation

#### Linux (Ubuntu/Debian)

1.  **Install build tools and CMake:**
    ```bash
    sudo apt-get update
    sudo apt-get install build-essential cmake libopencv-dev
    ```
2.  **Download and extract LibTorch:**
    ```bash
    wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
    unzip libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
    ```
3.  **Download and install ONNX Runtime:**
    Download the appropriate pre-built binary from the [ONNX Runtime releases page](https://github.com/microsoft/onnxruntime/releases) and extract it.

#### Windows

1.  **Install Visual Studio:** Make sure to include the "Desktop development with C++" workload.
2.  **Install CMake:** Download and install from the [official website](https://cmake.org/download/).
3.  **Install vcpkg:**
    ```bash
    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpkg
    ./bootstrap-vcpkg.bat
    ```
4.  **Install dependencies using vcpkg:**
    ```bash
    ./vcpkg install libtorch opencv onnxruntime --triplet x64-windows
    ```

## Dataset Preparation

This project is designed to work with datasets where the image's filename is its label. We will use the [Large Captcha Dataset from Kaggle](https://www.kaggle.com/datasets/akashguna/large-captcha-dataset).

1.  **Download the dataset:** Download the dataset from the Kaggle link and extract it.
2.  **Organize the data:** The dataset contains `train` and `valid` folders. The project's data loader recursively finds images in the provided directories. You will need to update the paths in the `configs.json` file.

## Configuration

The training and model parameters can be configured in the `configs.json` file. Here are some of the key settings:

  * **`data`**:
      * `train_path`: Path to the training dataset folder.
      * `val_path`: Path to the validation dataset folder.
      * `charset`: The set of characters the model should recognize.
  * **`training`**:
      * `batch_size`: The number of samples per batch.
      * `epochs`: The total number of training epochs.
      * `learning_rate`: The initial learning rate for the optimizer.
  * **`paths`**:
      * `model_save`: Path to save the best trained model.
      * `log_dir`: Directory to save training logs.

## Building the Project

1.  Create a build directory:
    ```bash
    mkdir build && cd build
    ```
2.  Run CMake.
      * **On Linux:**
        ```bash
        cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch;/path/to/onnxruntime ..
        ```
      * **On Windows (with vcpkg):**
        ```bash
        cmake -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake ..
        ```
3.  Compile the project:
    ```bash
    cmake --build . --config Release
    ```

## Training

To start the training process, run the compiled executable with the `--train` flag:

```bash
./captcha_solver --train --config /path/to/configs.json
```

The model will start training, and the best-performing model will be saved to the path specified in `paths.model_save` in your `configs.json` file.

## Inference

To run inference on a single CAPTCHA image, use the `--infer` flag, followed by the path to the image:

```bash
./captcha_solver --infer /path/to/your/captcha.png --config /path/to/configs.json
```

The model will load the saved weights and predict the text in the image.

## Exporting to ONNX

You can convert the trained PyTorch model (`.pt` file) to the ONNX format using the `export_onnx.py` script:

```bash
python scripts/export_onnx.py /path/to/your/model.pt /path/to/output/model.onnx
```

This will create an `model.onnx` file that can be used for deployment with various inference engines.