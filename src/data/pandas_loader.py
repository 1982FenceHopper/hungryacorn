import pandas as pd

def load_data(csv_path=None, csv_content=None):
    datapoints = pd.read_csv(csv_path if csv_path else csv_content, encoding="utf-8", low_memory=False)
    datapoints = datapoints.drop(0)
    datapoints["price"] = datapoints["price"].astype("float")
    datapoints["date"] = pd.to_datetime(datapoints["date"])
    datapoints = datapoints.sort_values("date")
    datapoints["month"] = datapoints["date"].dt.to_period("M")
    datapoints = datapoints.groupby("month")["price"].mean()
    datapoints = datapoints.reset_index().set_index("month")
    return datapoints

def main():
    data = load_data()
    data["date"] = pd.to_datetime(data["date"])
    data["month"] = data["date"].dt.to_period("M")
    m_avg = data.groupby("month")["price"].mean()
    print(m_avg.head(20))
    
if __name__ == '__main__':
    main()