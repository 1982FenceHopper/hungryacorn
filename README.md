# Mother C1

`Quick note, this codebase is real messy, many apologies 😭`

**_PSA: THIS IS A HIGHLY EXPERIMENTAL MODEL, THIS SHOULD NOT BE USED IN CRITICAL APPLICATIONS, [SEE THIS FOR SIDE DETAILS](#Stuff-I-wanna-say)_**

Mother C1 is an LSTM Deep Neural Network that predicts the food prices of various countries suffering from food insecurity worldwide. One of it's most prominent (albeit experimental and highly alpha) features, is it's ability to forecast future prices currency-agnostically, eliminating the need to convert local currencies to a stable exchange (e.g. USD, EUR) prior to forecasting future values (See [[3]](#Notes) for a PS on this)

The model was largely trained on available data from the [Humantarian Data Exchange](https://data.humdata.org/), namely the [World Food Programme Food Pricing Data](https://www.wfp.org/) (See [[1]](#Notes) for details on where)

**This project depends on my OpenACHES server for training, [see here](https://github.com/1982FenceHopper/openaches)**

## Technologies Used

- PyTorch - The core of this project, used to make the LSTM model
- pandas - Used to process CSV data for the model
- numpy - Self-explanatory
- ONNXRuntime - Exporting the model for ease of use, anywhere

## Variants

Note that for all variants, Forecasting Horizon (how many months into the future will it forecast) is a division by 2 of the Sequence Length (Basically how much past data does the model take into consideration when forecasting, in months). As such, it is recommended to have lengths of data atleast twice that of the variants forecasting horizon length (or just make sure your data has enough series of time equal to it's training sequence length, note that it doesn't count for the lite variant). Note that Sequence Length is a dynamic axes, so you don't HAVE TO conform to the models training parameters. (See [[4]](#Notes) for details on the Forecasting Horizon)

### Mother-C1 Lite

Sequence Length [DYNAMIC]: 12 months
<br />
Forecasting Horizon: 12 months
<br />
Epoch: 100
<br />
Average RMSE Training Loss: 5.236049384571023
<br />
Average RMSE Testing Loss: 0.120842023948273

### Mother-C1 Capstone

Sequence Length [DYNAMIC]: 96 months
<br />
Forecasting Horizon: 48 months
<br />
Epoch: 200
<br />
Average RMSE Training Loss: 9.395890903311637
<br />
Average RMSE Testing Loss: 0.642018204829109

### Mother-C1 Destiny [PLANNED]

Sequence Length [DYNAMIC]: 288 months
<br />
Forecasting Horizon: 48 months
<br />
Epoch: 200

## Quick Start

### ONNXRuntime

ONNX models are present in the repository under `src/models`, which are based on the latest PyTorch checkpoint for each respective model (Lite and Capstone)

For data, the format is as follows

```
month           price
2024-01-15      217.89
2024-02-15      219.24
2024-03-15      198.23
...             ...
```

Note that the month datetime does not have to be exactly like this, but it needs to be in a format that is processable by either PyTorch (if running the model using the raw class module) or the ONNX model. pandas is the library used to process data, see `main.py` to see how data is processed. (See [[2]](#Notes) for details on the example data used). You may also use pandas to conform the data to this format.

ONNXRuntime Example

```py
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
    data = requests.get("http://127.0.0.1:2190/api/v1/hdx/fp?country=YEM") # Depends on OpenACHES, modify if you are using an external resource
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

if __name__ == "__main__":
    main()
```

### PyTorch Raw Module Class

`main.py` is set up properly to run the model straight from PyTorch, you just need to use `pip` to install all the necessary dependencies

```bash
pip install -r requirements.txt
```

From there, you can execute `main.py` and all should be golden :D (If not, feel free to open an issue)

## Notes

[[1]](#Notes): The WFP data was from the HDX Data Platform, [See this link](https://data.humdata.org/dataset/?organization=wfp&q=Food+Prices&sort=score+desc%2C+last_modified+desc&ext_page_size=25)
<br />
[[2]](#Notes): Raw CSV is at `src/data/csv/wfp_food_prices_nga.csv`, which is the [WFP food pricing data for Nigeria](https://data.humdata.org/dataset/wfp-food-prices-for-nigeria)
<br />
[[3]](#Notes): AgnosticScaler is basically a glorified min-max scaler, but it stores the relative differential parameters for each year, which so far I haven't seen sklearn's MinMaxScaler do (or it probably does and I just did the biggest placebo effect to myself, let me know if so via an issue please 😭)
<br />
[[4]](#Notes): Having a Forecasting Horizon greater than 48-months tend to be a little too difficult when it comes to LSTM models, particularly in the case of error amplification. Regardless of how much data you put, most neural networks just are not meant to predict for over 4 years. I mean, you theoretically can, but unless you have a really precise, clean and huge (I mean HUGE, talking beyond 5 GB here 😭) data corpus, it's simply not worth it. Usually, there is no need to predict for that long (even 4 years is pushing it, most parties tend to predict just a year or two in advance). Also, univariate time series datasets can only get you so far (don't worry, that's a warning for me too, I will soon add sentimental analysis weights to Mother-C1 via news articles and whatnot for better predictions)

## Stuff I wanna say

I have no clue how I made this, just randomly started studying AI/ML and decided to create this (still having trouble wrapping my head around tensor operations but we all start somewhere right?). I had used tools like ChatGPT/Perplexity to help me out, so take my project with a grain of sand. This is the first, full-fledged ML project I've done, so there is bound to be countless bugs and issues. If you find any, please don't hesitate to create an issue, and I'll try my best to respond to them.

## License

```
Copyright (C) 2024 1982FenceHopper

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.
```
