import numpy as np

DNA_MAP = {
    "A": [1,0,0,0],
    "T": [0,1,0,0],
    "G": [0,0,1,0],
    "C": [0,0,0,1]
}

MAX_LEN = 300   # dataset standard

def one_hot_encode(seq):
    encoded = []
    for base in seq:
        encoded.append(DNA_MAP.get(base, [0,0,0,0]))
    return np.array(encoded)

def preprocess_dataset(dataset):
    X, y = [], []

    for item in dataset:
        seq = item["sequence"][:MAX_LEN]
        seq = seq.ljust(MAX_LEN, "A")   # padding
        X.append(one_hot_encode(seq))
        y.append(item["label"])

    return np.array(X), np.array(y)
