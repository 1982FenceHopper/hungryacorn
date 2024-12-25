import numpy as np
import onnxruntime
import pandas as pd
import requests, io, os
import matplotlib.pyplot as plt
import matplotlib as mlt
from pyprojroot import here
from sklearn.preprocessing import MinMaxScaler

class AgnosticScaler():
    def __init__(self):
        self.year_params = {}
        
    def fit_transform(self, df):
        df = df.copy()
        df["year"] = df.index.to_timestamp().year
        for year, group in df.groupby("year"):
            min_val = group["price"].min()
            max_val = group["price"].max()
            self.year_params[year] = {"min": min_val, "max": max_val}
            df.loc[group.index, 'agn_price'] = (group['price'] - min_val) / (max_val - min_val)
        return df["agn_price"]
    
    def inverse_transform(self, agn_vals, years):
        original = []
        max_year = max(self.year_params.keys())
        
        for agn_val, year in zip(agn_vals, years):
            if year in self.year_params:   
                params = self.year_params[year]
            else:
                params = self.year_params[max_year]
            original_val = agn_val * (params["max"] - params["min"]) + params["min"]
            original.append(original_val)
        return np.array(original)

def create_sequences(data: any, seq_length: int):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(x), np.array(y)

def main():
    
    data = requests.get("http://127.0.0.1:2190/api/v1/hdx/fp?country=YEM")
    csv = pd.read_csv(io.StringIO(data.text), low_memory=False)
    
    datapoints = csv.copy()
    
    datapoints = datapoints.drop(0)
    datapoints["price"] = datapoints["price"].astype("float")
    datapoints["date"] = pd.to_datetime(datapoints["date"])
    datapoints = datapoints.sort_values("date")
    datapoints["date"] = datapoints["date"].dt.to_period("M")
    datapoints = datapoints.groupby("date")["price"].mean()
    datapoints = datapoints.reset_index().set_index("date")
    
    original = datapoints.copy(deep=True)
    original = original.reset_index()
    original["date"] = original["date"].dt.strftime("%Y-%m-15")
    original["date"] = pd.to_datetime(original["date"])
    
    session = onnxruntime.InferenceSession(os.path.join(here(), "./testing/models/mother_c1_capstone.onnx"))
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    all_predictions = pd.DataFrame()
    prev = datapoints.copy(deep=True)
    
    for i in range(2):
        print(f"Predicting Time Cycle {i+1} of 4...")
        
        last_date = datapoints.index[-1].to_timestamp()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq="ME")
        
        agn_scaler, mm_scaler = AgnosticScaler(), MinMaxScaler()
        
        datapoints["price"] = agn_scaler.fit_transform(datapoints)
        scaled = mm_scaler.fit_transform(datapoints[["price"]])
        x, y = create_sequences(scaled, 96)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        
        x = np.array(x, dtype=np.float32)
        
        results = session.run([output_name], {input_name: x})
        
        predictions = mm_scaler.inverse_transform(results[0])
        predictions = agn_scaler.inverse_transform(predictions, future_dates.year)
        predictions = predictions.flatten()
        
        pred_df = pd.DataFrame({"date": future_dates, "price": predictions})
        pred_df["date"] = pred_df["date"].dt.to_period("M")
        pred_df = pred_df.set_index("date")
        
        prev = pd.concat([prev, pred_df], axis=0)
        prev.index = pd.PeriodIndex(prev.index, freq="M")
        
        all_predictions = pd.concat([all_predictions, pred_df], axis=0)
        all_predictions.index = pd.PeriodIndex(all_predictions.index, freq="M")
        
        datapoints = prev.copy(deep=True)
        datapoints.index = pd.PeriodIndex(datapoints.index, freq="M")
    
    # last_date = datapoints.to_timestamp(copy=True).index[-1]
    # future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq="ME")
    
    # agn_scaler, mm_scaler = AgnosticScaler(), MinMaxScaler()
    
    # datapoints["price"] = agn_scaler.fit_transform(datapoints)
    # scaled = mm_scaler.fit_transform(datapoints[["price"]])
    # x, y = create_sequences(scaled, 48)
    # x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    
    # x = np.array(x, dtype=np.float32)
    
    # results = session.run([output_name], {input_name: x})
    
    # predictions = mm_scaler.inverse_transform(results[0])
    # predictions = agn_scaler.inverse_transform(predictions, future_dates.year)
    # predictions = predictions.flatten()
    
    # pred_df = pd.DataFrame({"date": future_dates, "price": predictions})
    # pred_df["date"] = pred_df["date"].dt.strftime("%Y-%m-15")
    # pred_df["date"] = pd.to_datetime(pred_df["date"])
    
    all_predictions = all_predictions.reset_index()
    all_predictions["date"] = all_predictions["date"].dt.strftime("%Y-%m-15")
    all_predictions["date"] = pd.to_datetime(all_predictions["date"])
    
    original["price"] = original["price"].apply(lambda x: x / 130.43)
    all_predictions["price"] = all_predictions["price"].apply(lambda x: x / 130.43)
    
    mlt.use("pdf")
    fig, ax = plt.subplots()
    ax.plot(original["date"], original["price"], label="Original Data (HDX)")
    ax.plot(all_predictions["date"], all_predictions["price"], label="Predictions (Mother C1 Capstone, two 48-month cycles)")
    ax.set_title("Food Prices for Haiti (HTI)")
    ax.set_xlabel("Date (YYYY-MM-15)")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    save_path = os.path.join(here(), "testing", "graphs", "predictions.pdf")
    fig.savefig(save_path)
    
    # print("Original Data:\n\n", original)
    # print("\nPredicted Data:\n\n", all_predictions)
    
if __name__ == "__main__":
    main()