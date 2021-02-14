"""
Useful functions for making pretty plots in matplotlib of accuracy of stock
predictions.

@author: Riley Smith
Created: 1-21-2021
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from utils import COLORS

def plot_predicted_close(ticker, pred, true, save_to=None):
    """
    Plots the predictions for the given ticker.

    Args:
    --ticker: The ticker predictions are being plotted for (for setting the title
        of the plot)
    --pred: An ndarray of the predicted stock values
    --true: An ndarray of the same size as pred with the true stock values
    --save_to: If None, just show the figure with plt.show(). If a string, save
        the figure to that string.
    """
    # Build figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

    # Generate x values as ordinary arange
    x = np.arange(pred.size)

    # Reverse pred and true so that most recent entries are last
    pred = pred[::-1]
    true = true[::-1]

    # Plot both predictions and true values on same axis
    ax.plot(x, pred, color='red')
    ax.plot(x, true, color='blue')

    # Set axis labels
    ax.set_ylabel('Predicted Close')

    # Build legend
    red_patch = mpatches.Patch(color='red', label='Actual close')
    blue_patch = mpatches.Patch(color='blue', label='Predicted close')
    plt.legend(handles=[red_patch, blue_patch])

    # Set title
    plt.title(ticker)

    # Save output
    if save_to:
        plt.savefig(save_to)
    else:
        plt.show(block=True)

def plot_all_predicted_close(results_dict, save_to=None):
    """
    Plots the predicted close for each ticker in the results dictionary as
    subplots in one figure.

    Args:
    --results_dict: A dictionary where each key is a ticker and each value is
        a dictionary containing the keys:
            -'v_pred' (predicted closing prices)
            -'v_true' (actual closing prices)
    --save_to: If None, just show the figure with plt.show(). If a string, save
        the figure to that path.
    """
    # Build figure
    num_tickers = len(results_dict.keys())
    nrows = 2 if num_tickers >= 4 else 1
    ncols = int(np.ceil(num_tickers / 2)) if num_tickers >= 4 else num_tickers
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows))
    reshaped_axs = axs.reshape(-1)

    for i, ticker in enumerate(results_dict):
        # Get pred, true
        pred = results_dict[ticker]['v_pred'][::-1] # Reverse array
        true = results_dict[ticker]['v_true'][::-1]

        # Set x values
        x = np.arange(pred.size)

        # Plot it
        reshaped_axs[i].plot(x, pred, color='red')
        reshaped_axs[i].plot(x, true, color='blue')

        # Set subplot title
        reshaped_axs[i].title.set_text(ticker)

    # Build legend
    red_patch = mpatches.Patch(color='red', label='Actual close')
    blue_patch = mpatches.Patch(color='blue', label='Predicted close')
    plt.legend(handles=[red_patch, blue_patch])

    # Optionally save figure
    if save_to:
        plt.savefig(save_to)
    else:
        plt.show(block=True)

def plot_direction_accuracy(accuracy_dict, save_to=None):
    """
    Takes a dictionary as input and plots the accuracy of direction predictions
    for each ticker in the dictionary (as a % of predictions that are correct
    in direction).

    Args:
    --accuracy_dict: A dictionary. Each key is a ticker and each value is the
        accuracy of direction predictions (% of correct predictions)
    """
    # Build figure
    num_bars = len(accuracy_dict.keys())
    figsize = ((6/5) * num_bars, 6)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # Get accuracy by ticker
    accuracy = accuracy_dict.values()

    # Make bar chart
    x = np.arange(len(accuracy))
    bars = ax.bar(x, accuracy, tick_label=list(accuracy_dict.keys()))
    ax.set_ylim(bottom=0, top=1)

    # Set values above bars
    for acc, ticker_bar, color in zip(accuracy, bars, COLORS):
        height = ticker_bar.get_height()
        ax.text(ticker_bar.get_x() + ticker_bar.get_width()/2., 1.05*height,
                f'{round(acc, 2)}',
                ha='center', va='bottom')
        ticker_bar.set_color(color)

    # Save it if save path is specified
    if save_to is not None:
        plt.savefig(save_to)
    else:
        plt.show(block=True)
    return
