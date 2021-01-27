"""
Useful functions and globals for use in other files.

@author: Riley Smith
Created: 1-20-2021
"""
import sklearn
from tqdm import tqdm

from data import StockDataset
import metrics as mymetrics

# Define list of tickers used elsewhere in project
TICKERS = ['MSFT', 'AAPL', 'JNJ', 'BA', 'JPM',
            'TSLA', 'PFE', 'WMT', 'BP', 'CCMP']

# Define list of named colors in matplotlib
color_list = [
    'dodgerblue', 'lightcoral', 'green', 'gold', 'mediumpurple',
    'peru', 'darkcyan', 'violet', 'orange', 'limegreen'
]

# Make generator to create endless cycle of colors
def color_gen():
    i = 0
    while True:
        i += 1
        i = i % 10
        yield color_list[i]

# Build color generator
COLORS = color_gen()

# Define a function to use a classifier to predict directions for each stock
def predict_direction(clf, tickers, **kwargs):
    """
    Use clf (an untrained classifier) to predict direction of change for validation
    data for each stock in 'tickers'. Pass additional keyword arguments to be
    used in building the stock datasets.

    Args:
    --clf: An untrained sklearn classifier
    --tickers: A list of tickers to use
    --kwargs: Additional arguments for the StockDataset class

    Returns:
    A dictionary where each key is a ticker in 'tickers' and each value is the
    accuracy for the predictions for that ticker.
    """
    results = {}
    for ticker in tqdm(tickers):
        # Build and split dataset
        ds = StockDataset(tickers=ticker, quiet=True, **kwargs)
        t_data, v_data, t_label, v_label = ds.split(label_field='Direction')
        # Clone classifier
        clf_clone = sklearn.base.clone(clf)
        # Fit classifier to data
        clf_clone.fit(t_data, t_label)
        # Predict and store results
        v_pred = clf_clone.predict(v_data)
        results[ticker] = mymetrics.direction_accuracy(v_label, v_pred)
    return results

# Define function to predict closing prices for a list of stocks
def predict_close(clf, tickers, **kwargs):
    """
    Use clf (an untrained classifier) to predict closing price for validation
    data for each stock in 'tickers'. Pass additional keyword arguments to be
    used in building the stock datasets.

    Args:
    --clf: An untrained sklearn regressor
    --tickers: A list of tickers to use
    --kwargs: Additional arguments for the StockDataset class

    Returns:
    A dictionary where each key is a ticker in 'tickers' and each value is itself
    as dictionary containing the following:
        -'v_pred': The predicted closing prices
        -'v_true': The actual closing prices
    (both as ndarrays).
    """
    results = {}
    for ticker in tqdm(tickers):
        # Build and split dataset
        ds = StockDataset(tickers=ticker, quiet=True, **kwargs)
        t_data, v_data, t_label, v_label = ds.split(label_field='Label')
        # Clone classifier
        clf_clone = sklearn.base.clone(clf)
        # Fit classifier to data
        clf_clone.fit(t_data, t_label)
        # Predict and store results
        v_pred = clf_clone.predict(v_data)
        results[ticker] = {
            'v_pred': v_pred,
            'v_true': v_label
        }
    return results
