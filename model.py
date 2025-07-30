#This program implements the model architecture for a CRNN with CBAM attention mechanism.
#The model architecture includes a feature extractor with residual blocks,
#a positional encoding layer, and a CRNN with bidirectional LSTM and Transformer layers.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#The CBAM (Convolutional Block Attention Module) implementation.
#This module applies both channel and spatial attention mechanisms to the input feature maps.
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    #Applies channel and spatial attention to the input tensor.
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_attention = torch.sigmoid(avg_out + max_out)
        x = x * channel_attention
        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = torch.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_attention
        return x

#The Residual Block implementation with CBAM attention.
#Each block consists of two convolutional layers, batch normalization, ReLU activation,
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.cbam = CBAM(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    #Forward pass through the residual block.
    #It applies the convolutional layers, batch normalization, ReLU activation,
    #and CBAM attention.
    #The output is the sum of the input and the processed features, followed by ReLU activation.
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out += identity
        out = self.relu(out)
        return out

#The Feature Extractor module that uses a series of residual blocks to extract features from the input images.
#It includes a convolutional layer, batch normalization, ReLU activation, and max pooling.
class FeatureExtractor(nn.Module):
    def __init__(self, dropout=0.1):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 128, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(128, 256, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(256, 512, stride=1, dropout=dropout)
        self.layer4 = self._make_layer(512, 512, stride=(1,1), dropout=dropout)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, None))

    def _make_layer(self, in_channels, out_channels, stride, dropout):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride, dropout))
        layers.append(ResBlock(out_channels, out_channels, 1, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adaptive_pool(x)
        return x

#The Positional Encoding module that adds positional information to the input features.
#This is useful for models that do not have a built in notion of order, like in Transformers.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

#The CRNN model that combines the feature extractor, positional encoding, and a CRNN architecture.
#It includes a feature extractor, a convolutional layer, a LSTM layer, a Transformer encoder,
#layer normalization, dropout, and a final classifier layer.
class CRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=512, attention_heads=8, num_layers=4, dropout=0.2):
        super(CRNN, self).__init__()
        self.feature_extractor = FeatureExtractor(dropout=dropout)
        
        self.feature_conv = nn.Conv2d(512, hidden_size // 2, kernel_size=1)
        
        self.rnn = nn.LSTM(
            hidden_size, 
            hidden_size, 
            num_layers=3, 
            bidirectional=True, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size * 2, 
            nhead=attention_heads, 
            dim_feedforward=hidden_size * 4, 
            dropout=dropout, 
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, vocab_size + 1)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LSTM, nn.GRU)):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

    #The forward method processes the input tensor through the feature extractor,
    #applies a convolutional layer, reshapes the features, passes them through a RNN layer,
    #applies a Transformer encoder, normalizes the output, applies dropout,
    #and classifies the output using a linear layer.
    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.feature_conv(features)
        
        b, c, h, w = features.size()
        features = features.view(b, c * h, w).permute(0, 2, 1)
        
        rnn_out, _ = self.rnn(features)
        
        transformer_out = self.transformer_encoder(rnn_out)
        
        transformer_out = self.layer_norm(transformer_out)
        transformer_out = self.dropout(transformer_out)
        output = self.classifier(transformer_out)
        
        return output