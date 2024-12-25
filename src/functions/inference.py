import torch
from torch import nn
from torch import optim
from models import LSTM_RNN_Model
from sklearn.preprocessing import MinMaxScaler
from functions import create_seq
import numpy as np
import pandas as pd

#? Inferencing results in target size = torch.Size([32]) and input size = torch.Size([sequence_length, 32]), so PyTorch warns about broadcasting.
#? This doesn't seem to be affecting it at all though, so it's fine for now.

def inference(checkpoint_path, sequence_length, data):
    from functions import AgnosticScaler
    loss_total = 0.0
    
    original = data.copy()
    last_date = data.to_timestamp(copy=True).index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=sequence_length, freq="ME")
    
    mm_scaler = MinMaxScaler()
    agn_scaler = AgnosticScaler()
    
    data["price"] = agn_scaler.fit_transform(data)
    
    scaled = mm_scaler.fit_transform(data[["price"]])
    x, y = create_seq(scaled, sequence_length)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    
    tens_x = torch.tensor(x, dtype=torch.float32).to("cuda:0")
    tens_y = torch.tensor(y, dtype=torch.float32).to("cuda:0")
    
    print("Tensor X Shape: ", tens_x.shape)
    print("Tensor Y Shape: ", tens_y.shape)
    
    infer_set = torch.utils.data.TensorDataset(tens_x, tens_y)
    infer_dl = torch.utils.data.DataLoader(infer_set, batch_size=32, shuffle=False)
    
    model = LSTM_RNN_Model(input_dim=1, hidden_dim=50, num_layers=4, output_dim=1).to("cuda:0")
    criterion = nn.MSELoss().to("cuda:0")
    optimizer = optim.Adam(model.to("cuda:0").parameters(), lr=0.001)
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    horizon = checkpoint['horizon']
    
    model.eval().to("cuda:0")
    
    predictions = np.empty((0, horizon))
    
    with torch.no_grad():
        for bx, by in infer_dl:
            bx, by = bx.to("cuda:0"), by.to("cuda:0")
            current_input = bx
            print("Current Input Shape: ", current_input.shape)
            
            batch_predictions = []
            for _ in range(horizon):
                output = model(current_input.float())
                if output.dim() == 2:
                    output = torch.unsqueeze(output, dim=-1)
                    
                predicted = output[:, -1, :].squeeze()
                
                batch_predictions.append(predicted)
                current_input = torch.cat((current_input[:, 1:, :], output[:, -1:, :]), dim=1)
            
            batch_predictions = torch.stack(batch_predictions)
            loss = torch.sqrt(criterion(batch_predictions, by.to("cuda:0")) + 1e-6)
            loss_total += loss.item()
            
            print(batch_predictions.shape)
            predictions = np.concatenate((predictions, batch_predictions.to("cpu").numpy().reshape(-1, horizon)))
            
    loss_total /= len(infer_dl)
    predictions = mm_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    predictions = agn_scaler.inverse_transform(predictions, future_dates.year)
    
    print(f"\nAverage Inference Loss (RMSE): {loss_total:.2f}\n")
    
    print(f"Original Data: {original}\n")
    
    for date, prediction in zip(future_dates, predictions):
        print(f"Predicted Price for Month {date.strftime('%Y-%m')}: {prediction:.2f}")