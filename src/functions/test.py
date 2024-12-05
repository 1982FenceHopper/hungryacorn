import numpy as np
import onnxruntime
import pandas as pd
from functions import create_seq
from sklearn.preprocessing import MinMaxScaler

def onnx_test(model_path, data):
    from functions import AgnosticScaler
    
    original = data.copy()
    last_date = data.to_timestamp(copy=True).index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq="ME")
    
    mm_scaler, agn_scaler = MinMaxScaler(), AgnosticScaler()
    
    data["price"] = agn_scaler.fit_transform(data)
    scaled = mm_scaler.fit_transform(data[["price"]])
    x, y = create_seq(scaled, 12)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    
    x = np.array(x, dtype=np.float32)
    
    session = onnxruntime.InferenceSession(model_path)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    results = session.run([output_name], {input_name: x})
    
    predictions = mm_scaler.inverse_transform(results[0])
    predictions = agn_scaler.inverse_transform(predictions, future_dates.year)
    
    print("Original Data:\n\n", original)
    for date, prediction in zip(future_dates, predictions.flatten()):
        print(f"Predicted Price for Month {date.strftime('%Y-%m')}: {prediction:.2f}")