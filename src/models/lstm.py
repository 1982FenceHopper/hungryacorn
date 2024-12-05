import torch.nn as nn
import torch

class LSTM_RNN_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):        
        super(LSTM_RNN_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.LSTM = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.GELU = nn.GELU(approximate="tanh")
        self.DROP = nn.Dropout(p=0.2)
        self.LIN = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        tens_h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        tens_c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.LSTM(x, (tens_h0.detach(), tens_c0.detach()))
        out = self.GELU(out)
        out = self.DROP(out)
        out = self.LIN(out[:, -1, :])
        return out