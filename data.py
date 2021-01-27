"""
Script for collecting data using pandas-datareader.

@author: Riley Smith
Date: 1-3-2021
"""
import numpy as np
import pandas_datareader.data as web
import pandas as pd
import sklearn
from sklearn import preprocessing
from tqdm import tqdm

TICKERS = ['MSFT', 'AAPL', 'JNJ', 'BA', 'JPM',
            'TSLA', 'PFE', 'WMT', 'BP', 'CCMP']

##############################################
#  Useful functions for data pre-processing  #
##############################################

def add_derivative_stats(data, delta):
    """
    Takes a Pandas df of stock data and adds new columns with derivative stats.

    Args:
    --data: A Pandas df containing daily stock data
    --delta: None or an integer. If an integer, each value is adjusted to be a
        percentage change over that many days. If None, values are left as is.

    Returns:
    A Pandas df with 7 added columns
    """
    # Compute high minus low and open minus close and high minus open
    if delta is None:
        data['H-L'] = data.apply(lambda x: x['High'] - x['Low'], axis=1)
        data['O-C'] = data.apply(lambda x: x['Open'] - x['Close'], axis=1)
        data['H-O'] = data.apply(lambda x: x['High'] - x['Open'], axis=1)

    # Compute averages (use 5 trading days per week, not 7)
    if delta is None:
        data['7 Day Avg'] = data['Close'][::-1].rolling(5).mean()
        data['14 Day Avg'] = data['Close'][::-1].rolling(10).mean()
        data['21 Day Avg'] = data['Close'][::-1].rolling(15).mean()
    else:
        data['7 Day Change'] = (data['Close'] - data['Close'].shift(-5)) / data['Close'].shift(-5)
        data['7 Day Change'] = (data['Close'] - data['Close'].shift(-10)) / data['Close'].shift(-10)
        data['7 Day Change'] = (data['Close'] - data['Close'].shift(-15)) / data['Close'].shift(-15)

    # Compute 7 day standard deviation
    data['7 Day Std'] = data['Close'][::-1].rolling(5).std()

    return data

def retrieve_data(ticker, delta, label_horizon, add_derivatives=False):
    """
    Function to use Pandas DataReader and Stooq to retrieve data for the
    specified ticker and add the ticker as a column (so later data from multiple
    tickers can be appended into one dataframe). Also add a "Label" column which
    is the next day's closing price.

    Args:
    --ticker: The ticker to retrieve data from
    --delta: If an integer, make each feature a % change since that many days
        ago
    --add_derivatives: Whether or not to add derivative stats

    Returns:
    A Pandas dataframe with daily data for the specified ticker.
    """
    # Fetch data
    data = web.DataReader(ticker, 'stooq')
    # Optionally add derivative stats
    if add_derivatives:
        data = add_derivative_stats(data, delta)
    # Add label (for closing price)
    data['Label'] = data['Close'].shift(label_horizon)
    # Add direction indicator
    data['Direction'] = data.apply(lambda x: 1 if x['Label'] > x['Close'] else 0, axis=1)
    # Make % change if specified
    if delta:
        # Reformat label as percent change
        data['Label'] = (data['Label'] - data['Close']) / data['Close']
        for field in ['Open', 'Close', 'High', 'Low']:
            data[field] = (data[field] - data[field].shift(-1 * delta)) / data[field].shift(-1 * delta)
    # Add ticker column
    data['Ticker'] = [ticker] * len(data.index)
    return data

class StockDataset():
    def __init__(self, tickers=TICKERS, keep_volume=True, add_derivatives=True,
                    separate=True, quiet=False, delta=None, label_horizon=1):
        """
        Args:
        --tickers: A list of tickers to use or a string of just one ticker.
            Defaults to the 10 tickers in the global list TICKERS above
        --keep_volume: If True, volume is kept in data. If False, volume is
            discarded
        --add_derivatives: Whether or not to add derivative statistics to data
        --separate: If True, data for each stock is stored in a separate df.
            If False, there is just one master dataframe.
        --quiet: If True, does not print status updates as dataset is built
            (helpful for building dataset in another function or method that
            has its own status updates)
        --delta: If an integer, makes each feature a percentage change since
            that many days prior (i.e. delta=1 means each features i % change
            from previous day)
        --label_horizon: The number of trading days in the future to make the
            label
        """
        # Make tickers a list if they are a string
        if isinstance(tickers, str):
            tickers = [tickers]
        self.tickers = tickers

        # Load data in memory
        # NOTE: For using much larger data, a more sophistocated method may be
        # needed to not hold it in memory (here it is under 1mb so we're good)
        self.data = self._load_data(keep_volume, add_derivatives, quiet, delta, label_horizon)

        # Set dataset length
        self.length = len(self.data.index)

    def _load_data(self, keep_volume, add_derivatives, quiet, delta, label_horizon):
        """Load data and calculate derivative stats"""
        if not quiet:
            print('Initializing data...')
        # Fetch data
        if quiet:
            data = pd.concat([retrieve_data(ticker, delta, label_horizon,
                                add_derivatives=add_derivatives) for ticker in self.tickers], axis=0)
        else:
            data = pd.concat([retrieve_data(ticker, delta, label_horizon,
                                add_derivatives=add_derivatives) for ticker in tqdm(self.tickers)], axis=0)
        # Drop rows with NaN values
        data = data.dropna()
        # Optionally drop volume
        if not keep_volume:
            data = data.drop('Volume', axis=1)
        if not quiet:
            print('Done.')
        return data

    def split(self, train_size=0.8, split_mode='chronological', scale=False,
                label_field='Label'):
        """
        Custom train/test split method. Returns tuple of train data, test data,
        train label, and test label.

        Args:
        --train_size: A float between 0 and 1. The proportion of the dataset
            to use for training data
        --split_mode: One of 'chronological' or 'shuffle'. If 'chronological',
            uses the most recent data for test data. If 'shuffle', randomly
            splits data.
        --scale: Whether or not to preprocess the data with sklearn scale
        --label_field: The name of the column to use as a label (must be one of
            'Label', for closing price, or 'Direction', for the direction of
            change)

        Returns:
        A tuple with (in order) train data, test data, train label, test label
        (all as ndarrays).
        """
        # Extract labels and remove extraneous columns
        label = self.data[label_field]
        data = self.data.drop(['Ticker', 'Label', 'Direction'], axis=1)
        # Scale data
        if scale:
            data = preprocessing.scale(data, axis=1)

        if split_mode == 'shuffle':
            return sklearn.model_selection.train_test_split(data, label,
                                            train_size=train_size,
                                            shuffle=True, random_state=1234)
        elif split_mode == 'chronological':
            val_samples = round((1 - train_size) * self.length)
            t_data = data[val_samples:].to_numpy(dtype=np.float32)
            t_label = label.iloc[val_samples:].to_numpy(dtype=np.float32)
            v_data = data[:val_samples].to_numpy(dtype=np.float32)
            v_label = label.iloc[:val_samples].to_numpy(dtype=np.float32)
            return t_data, v_data, t_label, v_label

    def batch(self, shuffle=True, batch_size=16, random_seed=None):
        """
        Build a generator for presenting the data in batches.

        Args:
        --shuffle: If True, shuffles the dataset before batching
        --batch_size: The batch size
        --random_seed: Optional seed to seed the Numpy random number generator
            (so results can be reproduced)

        Returns:
        Generator for batching the data.
        """
        # Seed the random number generator
        if random_seed is not None:
            np.random.seed(random_seed)

        indices = np.arange(self.length)

        if shuffle:
            np.random.shuffle(indices)

        # Extract batches
        for i in range(0, self.length, batch_size):
            batch_indices = indices[i: i + batch_size]
            yield self.data.iloc[batch_indices].drop('Ticker', axis=1).to_numpy(dtype=np.float32) # NOTE: Ticker excluded from batch

    def drop(self, fields):
        """Drop the given fields from the data"""
        if isinstance(fields, str):
            fields = [fields]
        self.data.drop(fields, axis=1)
        return
