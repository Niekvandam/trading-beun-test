import pandas as pd

def walk_forward_splits(data, n_splits=4, train_size=0.7):
    """
    Split the data into multiple train/test sets in a walk-forward manner.
    For example, if n_splits=4, you get 4 periods of optimization+test.
    train_size is the fraction of each segment used for training.
    """
    length = len(data)
    segment_size = length // n_splits
    folds = []
    
    for i in range(n_splits):
        segment = data.iloc[i*segment_size:(i+1)*segment_size]
        train_len = int(len(segment)*train_size)
        train_data = segment.iloc[:train_len]
        test_data = segment.iloc[train_len:]
        folds.append((train_data, test_data))
    return folds
