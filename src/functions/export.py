from pyprojroot import here
import os
import torch
from models import LSTM_RNN_Model

def onnx_export(checkpoint_path):
    dummy_tens = torch.randn((1, 12, 1)).to("cuda:0")
    
    model = LSTM_RNN_Model(input_dim=1, hidden_dim=50, num_layers=4, output_dim=1).to("cuda:0")
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])    
    
    model.eval().to("cuda:0")
    
    filepath = os.path.join(here(), 'src', 'models', 'lstm_latest.onnx')
    
    torch.onnx.export(
        model,
        dummy_tens,
        filepath,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size", 1: "sequence_length"}
        }
    )
    
    print(f"Model successfully exported to {filepath}")