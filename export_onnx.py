#This program exports a trained CRNN model to ONNX format for deployment.
#It requires a configuration file and the trained model weights to be present.
#Usage: python export_onnx.py --config config.json --output captcha_solver.onnx

import torch
import argparse
import os
from model import CRNN
from utils import load_config

#Load configuration from a JSON file.
def export_to_onnx(config, output_path):
    device = torch.device('cpu')
    
    #Check if the output directory exists, create it if not.
    print("Initializing model...")
    model = CRNN(
        vocab_size=len(config['data']['charset']), 
        hidden_size=config['model']['hidden_size'], 
        attention_heads=config['model']['attention_heads'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    )
    
    model_save_path = config['paths']['model_save']
    try:
        print(f"Loading trained weights from {model_save_path}...")
        state_dict = torch.load(model_save_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_save_path}. Please train the model first.")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    #Ensure the output directory exists.
    model.eval()

    #Create a dummy input tensor based on the configuration.
    dummy_input = torch.randn(
        1, 
        3, 
        config['data']['image_height'], 
        config['data']['image_width'],
        device=device
    )

    try:
        print(f"Exporting model to ONNX format at {output_path}...")
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"Model successfully exported to {output_path}.")
    except Exception as e:
        print(f"Error during ONNX export: {e}")

if __name__ == "__main__":
    #Parse command line arguments for configuration and output paths.
    parser = argparse.ArgumentParser(description="Export CRNN model to ONNX format")
    #Add arguments for configuration file and output path.
    parser.add_argument('--config', type=str, default="config.json", help="Path to the training config file.")
    parser.add_argument('--output', type=str, default="captcha_solver.onnx", help="Path to save the ONNX model.")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        export_to_onnx(config, args.output)
    except Exception as e:
        print(f"An error occurred: {e}")
