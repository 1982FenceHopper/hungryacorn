# HungryAcorn

`thats a temporary name i swear 😭`

**_PSA: THIS IS A HIGHLY EXPERIMENTAL MODEL, THIS SHOULD NOT BE USED IN CRITICAL APPLICATIONS, [SEE THIS FOR SIDE DETAILS](#Stuff-I-wanna-say)_**

HungryAcorn is an LSTM Deep Neural Network that predicts the food prices of various countries suffering from food insecurity worldwide. One of it's most prominent (albeit experimental and highly alpha) features, is it's ability to forecast future prices currency-agnostically, eliminating the need to convert local currencies to a stable exchange (e.g. USD, EUR) prior to forecasting future values (See [[3]](#Notes) for a PS on this)

The model was largely trained on available data from the [Humantarian Data Exchange](https://data.humdata.org/), namely the [World Food Programme Food Pricing Data](https://www.wfp.org/) (See [[1]](#Notes) for details on where)

**This project depends my OpenACHES server, which is another project of mine and will be posted on a repo soon**

## Technologies Used

- PyTorch - The core of this project, used to make the LSTM model
- pandas - Used to process CSV data for the model
- numpy - Self-explanatory
- ONNXRuntime - Exporting the model for ease of use, anywhere

## Quick Start

### ONNXRuntime

An ONNX model is present in the repository under `src/models/lstm_latest.onnx`, which is the ONNX model based on the current latest PyTorch checkpoint `src/models/checkpoints/2024-12-04T22_16_01_lstm.pth` (NOTE: Name of the checkpoint starts with an ISO 8601 date, ':' and '+' are formated into '\_' for filename compatiblity)

For data, the format is as follows

```
month           price
2024-01-15      217.89
2024-02-15      219.24
2024-03-15      198.23
...             ...
```

Note that the month datetime does not have to be exactly like this, but it needs to be in a format that is processable by either PyTorch (if running the model using the raw class module) or the ONNX model. pandas is the library used to process data, see `main.py` to see how data is processed. (See [[2]](#Notes) for details on the example data used). You may also use pandas to conform the data to this format.

ONNXRuntime Python Example (from `src/functions/test.py`)

```py
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
```

### PyTorch Raw Module Class

`main.py` is set up properly to run the model straight from PyTorch, you just need to use `pip` to install all the necessary dependencies

```bash
pip install -r requirements.txt
```

From there, you can execute `main.py` and all should be golden :D (If not, feel free to open an issue)

## Short Metrics

Average loss

## Notes

[[1]](#Notes): The WFP data was from the HDX Data Platform, [See this link](https://data.humdata.org/dataset/?organization=wfp&q=Food+Prices&sort=score+desc%2C+last_modified+desc&ext_page_size=25)
<br />
[[2]](#Notes): Raw CSV is at `src/data/csv/wfp_food_prices_nga.csv`, which is the [WFP food pricing data for Nigeria](https://data.humdata.org/dataset/wfp-food-prices-for-nigeria)

[[3]](#Notes): AgnosticScaler is basically a glorified min-max scaler, but it stores the relative differential parameters for each year, which so far I haven't seen sklearn's MinMaxScaler do (or it probably does and I just did the biggest placebo effect to myself, let me know if so via an issue please 😭)

## Stuff I wanna say

I have no clue how I made this, just randomly started studying AI/ML and decided to create this (still having trouble wrapping my head around tensor operations but we all start somewhere right?). I had used tools like ChatGPT/Perplexity to help me out, so take my project with a grain of sand. This is the first, full-fledged ML project I've done, so there is bound to be countless bugs and issues. If you find any, please don't hesitate to create an issue, and I'll try my best to respond to them.

## License

### Copyright 2024, Nashat Yafi (1982FenceHopper)

### This project is licensed under GNU AGPLv3, view `LICENSE` for more details.
