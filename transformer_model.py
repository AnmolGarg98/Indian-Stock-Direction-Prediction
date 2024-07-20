import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
sequence_length = 12
input_size = 7
output_size = 1
num_layers = 2
d_model = 64
nhead = 4
num_epochs = 50
learning_rate = 0.001

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, d_model, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.input_fc = nn.Linear(input_size, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.output_fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_fc(x)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, d_model)
        # print(x.shape)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.output_fc(x)
        return x[:, -1, :]