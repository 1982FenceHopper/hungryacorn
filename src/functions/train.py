from pyprojroot import here
import os
import torch
from torch import nn
from torch import optim
from models import LSTM_RNN_Model
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from functions import create_seq
import numpy as np
from datetime import datetime, timezone

def train(checkpoint_path, sequence_length, data):
    from functions import AgnosticScaler
    loss_total = 0.0
    
    mm_scaler = MinMaxScaler()
    agn_scaler = AgnosticScaler()
    
    data["price"] = agn_scaler.fit_transform(data)
    
    scaled = mm_scaler.fit_transform(data[["price"]])
    x, y = create_seq(scaled, sequence_length)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    
    tens_x = torch.tensor(x, dtype=torch.float32).to("cuda:0")
    tens_y = torch.tensor(y, dtype=torch.float32).to("cuda:0")
    
    train_set = torch.utils.data.TensorDataset(tens_x, tens_y)
    train_dl = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=False)
    
    model = LSTM_RNN_Model(input_dim=1, hidden_dim=50, num_layers=4, output_dim=1).to("cuda:0")
    criterion = nn.MSELoss().to("cuda:0")
    optimizer = optim.Adam(model.to("cuda:0").parameters(), lr=0.001)
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    model.train().to("cuda:0")
    
    torch.autograd.set_detect_anomaly(True)
    for epoch in tqdm(range(epoch), desc="Epoch"):
        for bx, by in train_dl:
            optimizer.zero_grad()
            bx, by = bx.to("cuda:0"), by.to("cuda:0")
            y_pred = model(bx)
            loss = torch.sqrt(criterion(y_pred.squeeze(), by) + 1e-6)
            loss_total += loss.item()
            loss.backward()
            optimizer.step()
            
    loss_total /= len(train_dl)
    print(f"Average Training Loss (RMSE): {loss_total}")
    
    current_time = datetime.now(timezone.utc)
    filename = f"{current_time.isoformat().replace(':', '_').replace('+', '_').split('.')[0]}_lstm.pth"
    
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": loss_total,
    }, os.path.join(here(), "src", "models", "checkpoints", filename))
    
    print(f"Model Checkpoint Saved To: {os.path.join(here(), 'src', 'models', 'checkpoints', filename)}")