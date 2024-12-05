import numpy as np

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