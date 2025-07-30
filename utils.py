#This is a utility module for a CAPTCHA recognition model.
#It includes functions for loading configurations, decoding CTC outputs,
#calculating metrics, and setting up logging.

import json
import logging
import os
import torch
import torch.nn.functional as F
import editdistance
from sklearn.metrics import accuracy_score

#Load configuration from a JSON file.
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

#Decode CTC outputs to text using a character set.
#The charset should include all characters used in the CAPTCHA, plus a blank character.
class CTCDecoder:
    def __init__(self, charset):
        self.charset = charset
        self.blank_idx = len(charset)

    #Decode the predicted indices from the model output.
    #The predictions are expected to be in log softmax format.
    def __call__(self, preds):
        pred_t = F.log_softmax(preds, dim=2).argmax(dim=2)
        pred_t = pred_t.detach().cpu().numpy()

        decoded_texts = []
        for P in pred_t:
            decoded = []
            for i, index in enumerate(P):
                if index != self.blank_idx:
                    if i == 0 or P[i-1] != index:
                        decoded.append(self.charset[index])
            decoded_texts.append("".join(decoded))
        return decoded_texts

#Metrics for evaluating the model's performance.
#Includes character error rate, sequence accuracy, and character accuracy.
class Metrics:
    @staticmethod
    def character_error_rate(preds, targets):
        total_dist = sum(editdistance.eval(p, t) for p, t in zip(preds, targets))
        total_len = sum(len(t) for t in targets)
        return total_dist / total_len if total_len > 0 else 0

    @staticmethod
    def sequence_accuracy(preds, targets):
        correct = sum(1 for p, t in zip(preds, targets) if p == t)
        return correct / len(targets) if len(targets) > 0 else 0

    @staticmethod
    def character_accuracy(preds, targets):
        total_correct = 0
        total_chars = 0
        for pred, target in zip(preds, targets):
            dist = editdistance.eval(pred, target)
            total_correct += max(0, len(target) - dist)
            total_chars += len(target)
        return total_correct / total_chars if total_chars > 0 else 0

#Setup a logger for the training process.
#It logs messages to both a file and the console.
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('CaptchaLogger')
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'), mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger