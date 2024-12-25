from data import load_data
from models import LSTM_RNN_Model
from functions import train, create_seq, inference, onnx_export, onnx_test

import requests
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os, sys, io
from pyprojroot import here
from datetime import datetime, timezone
from time import time

SEQ_LENGTH = 96
NUM_EPOCHS = 200
HORIZON = 48

NGA_FP_CSV = os.path.join(here(), 'src', 'data', 'csv', 'wfp_food_prices_nga.csv')

def main():
    from functions import AgnosticScaler
    print("Loading Data...")
    datapoints = load_data(NGA_FP_CSV)
    original = datapoints.copy()
    last_date = datapoints.to_timestamp(copy=True).index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=HORIZON, freq="ME")
    
    print("\nCreating Sequences...")
    
    mm_scaler = MinMaxScaler()
    agn_scaler = AgnosticScaler()
    
    datapoints["price"] = agn_scaler.fit_transform(datapoints)
    
    scaled = mm_scaler.fit_transform(datapoints[["price"]])
    x, y = create_seq(scaled, SEQ_LENGTH)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    
    print("\nCreating tensors...")
    tens_x = torch.tensor(x, dtype=torch.float32).to("cuda:0")
    tens_y = torch.tensor(y, dtype=torch.float32).to("cuda:0")
    
    train_size = int(0.8 * len(tens_x))
    
    x_train, x_test = tens_x[:train_size], tens_x[train_size:]
    y_train, y_test = tens_y[:train_size], tens_y[train_size:]
    
    train_dts, test_dts = TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)
    train_loader, test_loader = DataLoader(train_dts, batch_size=32, shuffle=True), DataLoader(test_dts, batch_size=32, shuffle=False)
    
    print("\nCreating Model...")
    model = LSTM_RNN_Model(input_dim=1, hidden_dim=50, num_layers=4, output_dim=1).to("cuda:0")
    criterion = nn.MSELoss().to("cuda:0")
    optimizer = optim.Adam(model.to("cuda:0").parameters(), lr=0.001)
    
    curr_loss = 0.0
    train_loss = 0.0
    
    print("\nTraining Model...")
    torch.autograd.set_detect_anomaly(True)
    model.train().to("cuda:0")
    for epoch in tqdm(range(NUM_EPOCHS), desc="Epoch"):
        for bx, by in train_loader:
            optimizer.zero_grad()
            bx, by = bx.to("cuda:0"), by.to("cuda:0")
            out = model(bx)
            curr_loss = torch.sqrt(criterion(out.squeeze(), by).to("cuda:0") + 1e-6).to("cuda:0")
            train_loss += curr_loss
            curr_loss.backward()
            optimizer.step()
            
    print("\nTesting Model...")
    model.eval().to("cuda:0")
    
    test_loss = 0.0
    predictions = []
    # current_input = x_test[-1].reshape(1, SEQ_LENGTH, 1).to("cuda:0")
    
    # for _ in tqdm(range(HORIZON)):
    #     with torch.no_grad():
    #         predicted = model(torch.tensor(current_input).float()).item()
    #         curr_test_loss = torch.sqrt(criterion(torch.tensor(predicted).to("cuda:0").float(), y_test[-1]) + 1e-6).to("cuda:0")
    #         test_loss += curr_test_loss
            
    #         predictions.append(predicted)
    #         pred_tens = torch.tensor([[[predicted]]]).to("cuda:0")
    #         current_input = torch.cat((current_input[:, 1:, :], pred_tens), dim=1)
    
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to("cuda:0"), by.to("cuda:0")
            current_input = bx
            
            batch_predictions = []
            for _ in range(HORIZON):
                output = model(current_input.float())
                if output.dim() == 2:
                    output = torch.unsqueeze(output, dim=-1)
                    
                predicted = output[:, -1, :].squeeze()
                
                batch_predictions.append(predicted)
                current_input = torch.cat((current_input[:, 1:, :], output[:, -1:, :]), dim=1)
                
            batch_predictions = torch.stack(batch_predictions)
            loss = torch.sqrt(criterion(batch_predictions, by.to("cuda:0")) + 1e-6)
            test_loss += loss.item()
            
            predictions.extend(batch_predictions.to("cpu").numpy().reshape(-1, HORIZON))
            
    predictions_inversed = mm_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    localized_predictions = agn_scaler.inverse_transform(predictions_inversed, future_dates.year)
    
    print(f"Original Data:\n{original}")
    
    for date, prediction in zip(future_dates, localized_predictions):
        print(f"Predicted Price for Month {date.strftime('%Y-%m')}: {prediction:.2f}")
     
    avg_train_loss = train_loss / len(train_loader)
    print(f"\nAverage Train Loss: {avg_train_loss:.4f}")        
    avg_loss = test_loss / len(test_loader)
    print(f"\nAverage Test Loss: {avg_loss:.4f}")
    
    current_time = datetime.now(timezone.utc)
    
    torch.save({
        "epoch": NUM_EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": avg_train_loss,
        "test_loss": avg_loss,
        "horizon": HORIZON
    }, os.path.join(here(), "src", "models", "checkpoints_capstone", f"{current_time.isoformat().replace(':', '_').replace('+', '_').split('.')[0]}_capstone.pth"))
    
    del model, criterion, optimizer
    torch.cuda.empty_cache()
            
    
if __name__ == '__main__':
    options = {
        0: "train",
        1: "inference",
        2: "export",
        3: "test",
        4: "initialize",
        5: "exit"
    }
    
    init = input("Choose an option:\n[0] Train\n[1] Inferencing\n[2] Export\n[3] Test ONNX Model\n[4] Initialize\n[5] Exit\n\n> ")
    if int(init) not in options:
        print("Invalid option.")
        exit()
    else:
        if options[int(init)] == "exit":
            exit()
        elif options[int(init)] == "train":
            print("\nTraining Mode\n")
            sel_chkt = input("Enter relative path to checkpoint: ")
            if not os.path.exists(os.path.join(here(), sel_chkt)):
                print("Checkpoint does not exist.")
                exit()
            country = input("Enter country: ")
            print("Retrieving dataset...")
            req = requests.get(f"http://localhost:2190/api/v1/hdx/fp?country={country}")
            csv_data = req.text
            csv_filename = req.headers["ORIGIN-FILENAME"]
            print("Augmenting data...")
            data = load_data(csv_content=io.StringIO(csv_data))
            print("Training selected checkpoint...")
            train(sel_chkt, 96, data)
        elif options[int(init)] == "inference":
            print("\nEvaluation Mode\n")
            sel_chkt = input("Enter relative path to checkpoint: ")
            if not os.path.exists(os.path.join(here(), sel_chkt)):
                print("Checkpoint does not exist.")
                exit()
            country = input("Enter country: ")
            print("Retrieving dataset...")
            req = requests.get(f"http://localhost:2190/api/v1/hdx/fp?country={country}")
            csv_data = req.text
            csv_filename = req.headers["ORIGIN-FILENAME"]
            print("Augmenting data...")
            data = load_data(csv_content=io.StringIO(csv_data))
            print("Evaluating selected checkpoint...")
            inference(sel_chkt, 96, data)
        elif options[int(init)] == "export":
            print("\nONNXRuntime Export Mode\n")
            sel_chkt = input("Enter relative path to checkpoint to export: ")
            if not os.path.exists(os.path.join(here(), sel_chkt)):
                print("Checkpoint does not exist.")
                exit()
            onnx_export(sel_chkt)
        elif options[int(init)] == "test":
            print("\nONNX Test Mode\n")
            sel_model = input("Enter relative path to ONNX model: ")
            if not os.path.exists(os.path.join(here(), sel_model)):
                print("Checkpoint does not exist.")
                exit()
            country = input("Enter country: ")
            print("Retrieving dataset...")
            req = requests.get(f"http://localhost:2190/api/v1/hdx/fp?country={country}")
            csv_data = req.text
            csv_filename = req.headers["ORIGIN-FILENAME"]
            print("Augmenting data...")
            data = load_data(csv_content=io.StringIO(csv_data))
            print("Testing selected ONNX model...")
            start = time()
            onnx_test(sel_model, data)
            end = time()
            print(f"\n\nONNX Test Time: {(end - start):.2f} seconds")
        elif options[int(init)] == "initialize":
            print("\nInitializing Model\n")
            main()
            
    
    # if (len(sys.argv) > 2):
    #     print("Retrieving dataset...")
    #     req = requests.get(f"http://localhost:2190/api/v1/fp?country={sys.argv[2]}")
    #     csv_data = req.text
    #     csv_filename = req.headers["ORIGIN-FILENAME"]
    #     print("Augmenting data...")
    #     data = load_data(csv_content=io.StringIO(csv_data))
    #     print("Training/Inferencing selected checkpoint...")
    #     inference(sys.argv[1], 12, data)
    # else:
    #     main()