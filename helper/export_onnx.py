import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

class STNLocalizationNet(nn.Module):
    def __init__(self):
        super(STNLocalizationNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 6)
        
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(identity)
        return F.relu(out)

class CNNBackbone(nn.Module):
    def __init__(self):
        super(CNNBackbone, self).__init__()
        self.stem = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.stem_bn = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 64, 3, 1)
        self.layer2 = self._make_layer(64, 128, 4, 2)
        self.layer3 = self._make_layer(128, 256, 6, 2)
        self.layer4 = self._make_layer(256, 512, 3, 2)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem(x)))
        x = F.max_pool2d(x, 3, 2, 1)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.w_o(context)

class CRNN(nn.Module):
    def __init__(self, vocab_size=63):
        super(CRNN, self).__init__()
        self.vocab_size = vocab_size
        
        self.stn = STNLocalizationNet()
        self.cnn = CNNBackbone()
        
        self.lstm1 = nn.LSTM(512, 256, num_layers=2, dropout=0.3, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, num_layers=2, dropout=0.3, bidirectional=True, batch_first=True)
        self.lstm3 = nn.LSTM(512, 128, num_layers=1, bidirectional=True, batch_first=True)
        
        self.lstm_dropout = nn.Dropout(0.3)
        self.attention = MultiHeadAttention(256, 8)
        self.classifier = nn.Linear(256, vocab_size)

    def forward(self, x):
        theta = self.stn(x)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        
        x = self.cnn(x)
        
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(batch_size, width, channels * height)
        
        x, _ = self.lstm1(x)
        x = self.lstm_dropout(x)
        
        x, _ = self.lstm2(x)
        x = self.lstm_dropout(x)
        
        x, _ = self.lstm3(x)
        
        x = self.attention(x)
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=2)

def export_to_onnx(pytorch_model_path, onnx_model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CRNN(vocab_size=63)
    checkpoint = torch.load(pytorch_model_path, map_location=device)
    
    if hasattr(checkpoint, 'state_dict'):
        model.load_state_dict(checkpoint.state_dict())
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(device)
    
    dummy_input = torch.randn(1, 3, 64, 256).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"Model successfully exported to {onnx_model_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python export_onnx.py <pytorch_model_path> <onnx_output_path>")
        sys.exit(1)
    
    pytorch_model_path = sys.argv[1]
    onnx_model_path = sys.argv[2]
    
    if not os.path.exists(pytorch_model_path):
        print(f"Error: PyTorch model file '{pytorch_model_path}' not found.")
        sys.exit(1)
    
    try:
        export_to_onnx(pytorch_model_path, onnx_model_path)
    except Exception as e:
        print(f"Error during export: {str(e)}")
        sys.exit(1)