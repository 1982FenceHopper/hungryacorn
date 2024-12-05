import numpy as np
from tqdm import tqdm

def create_seq(data: any, seq_length: int):
    x, y = [], []
    for i in tqdm(range(len(data) - seq_length), desc="Sequences"):
        x.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(x), np.array(y)