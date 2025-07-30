#This program evaluates a trained CRNN model on a set of CAPTCHA images.
#It loads the model, processes the images, and computes accuracy metrics.

import torch
from PIL import Image
import argparse
import os
import numpy as np
import warnings
from tqdm import tqdm

from model import CRNN
from utils import CTCDecoder, load_config, Metrics
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore", category=UserWarning)

#Define the image transformation for inference
#This includes resizing, normalization, and conversion to tensor format.
def get_inference_transform(height, width):
    return A.Compose([
        A.Resize(height, width),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

#Function to evaluate the model on a set of images
#It loads the model, processes each image, and computes sequence and character accuracy.
def evaluate(config, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    #Initialize the CRNN model with parameters from the config.
    model = CRNN(
        vocab_size=len(config['data']['charset']),
        hidden_size=config['model']['hidden_size'],
        attention_heads=config['model']['attention_heads'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    )

    #Load the model state from the specified path or from the config.
    model_path = args.model_path if args.model_path else config['paths']['model_save']

    try:
        loaded_object = torch.load(model_path, map_location=device, weights_only=True)
        state_dict = loaded_object.get('model_state_dict', loaded_object)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if os.path.isdir(args.input_path):
        image_paths = [os.path.join(args.input_path, fname) for fname in sorted(os.listdir(args.input_path))
                       if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(image_paths)} images in '{args.input_path}' for evaluation.")
    else:
        print(f"Error: Input path '{args.input_path}' is not a valid directory.")
        return

    if not image_paths:
        print("No images found to evaluate.")
        return

    transform = get_inference_transform(config['data']['image_height'], config['data']['image_width'])
    decoder = CTCDecoder(config['data']['charset'])

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="Evaluating Test Set"):
            try:
                target_label = os.path.splitext(os.path.basename(image_path))[0]
                
                image = Image.open(image_path).convert('RGB')
                image_np = np.array(image)
                image_tensor = transform(image=image_np)['image'].unsqueeze(0).to(device)

                output = model(image_tensor)
                prediction = decoder(output)[0]

                all_predictions.append(prediction)
                all_targets.append(target_label)
            except Exception as e:
                print(f"Skipping file {os.path.basename(image_path)} due to error: {e}")

    if not all_targets:
        print("Could not evaluate any images.")
        return
        
    seq_acc = Metrics.sequence_accuracy(all_predictions, all_targets)
    char_acc = Metrics.character_accuracy(all_predictions, all_targets)

    print("\nEvaluation Results:")
    print(f"Sequence Accuracy: {seq_acc * 100:.2f}%")
    print(f"Character Accuracy: {char_acc * 100:.2f}%")

if __name__ == "__main__":
    #Parse command line arguments for configuration and input paths.
    parser = argparse.ArgumentParser(description="Evaluate CAPTCHA model on a test set")
    #Add arguments for configuration file, input path, and model path.
    parser.add_argument('--config', type=str, default="config.json", help="Path to the config file")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the test image directory")
    parser.add_argument('--model_path', type=str, default=None, help="Path to a model or checkpoint file (overrides config)")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        evaluate(config, args)
    except FileNotFoundError:
        print(f"Error: Config file not found at '{args.config}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
