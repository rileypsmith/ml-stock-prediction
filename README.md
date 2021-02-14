# Stock Prediction with Basic ML

This repo is intended to investigate different machine learning methods in
Scikit-Learn applied to the problem of stock prediction.

## The problem

Predicting the closing price of stocks is a quintessential machine learning problem. This project tries to predict both the next day closing price of stocks and the next day direction of price change. The point of this project is to experiment with different Scikit-Learn methods and explore the relative advantages and disadvantages of each, not to accurately predict closing prices. None of the models presented here are accurate enough to trade with.

## Data

Data is taken from Stooq using Pandas-DataReader. The file data.py contains the class used for data in this project. In addition to providing a wrapper for actually fetching the data, the defined class (called `StockDataset()`) provides additional functionality, including a method to drop unwanted fields, the ability to add derivative stats to the data, and a choice of how far in the future to set the labels. While the analysis in this project only uses next day labels for closing price and price change of direction, the `StockDataset` class allows for looking at different windows of time.

## Analysis

Four common machine learning methods are used in this analysis. For each, a Jupyter notebook containing the used code is found in the `ml` folder. A brief discussion of the merits of each is found below

### K Nearest Neighbors

KNN is a simple classifier that finds the `k` closest data points in the training data to some test item (as defined by some distance metric, in this case Euclidean distance). It looks at the label on these `k` neighbors and uses those labels to predict the label for the test datapoint. In this analysis, both a `KNeighborsClassifier` and `KNeighborsRegressor` from `sklearn` are used, the classifier to predict directional change and the regressor to predict closing price.

KNN has several important advantages over other ML methods. For one, it requires no training. At test time, a point is just compared to a dataset of other points to find its K nearest neighbors, so all "training" takes place at test time. This also means that a KNN classifier does not need to be re-trained with the addition of new data. However, KNN can be very slow on large datasets. For the data used here, speed is not a problem, but KNN classifiers are also very sensitive to non-normalized data. For instance, try including the volume of the stock without normalization and see how this effects KNN results. You will notice that they become horribly inaccurate because the distance in the volume field far outweighs the distance in any other dimension.

Another key shortcoming of KNN is that it does not do well on out of sample predictions. Take a look at the predicted closing prices for Microsoft in `ml/knn.ipynb`. You'll notice that there is a perfectly horizontal line around a predicted value of $163. That is because the actual price of Microsoft at that time was higher than at any point since 2016, i.e. higher than any example in the training data. As such, the `k` nearest neighbors are the same for every point in that range, and all give the same prediction, namely the highest prediction they could possibly give (the average closing price of the `k` examples with the highest closing prices).

When calculating the Mean Average Percentage Error (MAPE), these out of sample predictions are removed, but still KNN does not get anything better than 2% MAPE on any stock. That sounds pretty good, but considering that a 2% daily change in a stock's price is a significant movement, it is actually poor to average 2% absolute error on closing price predictions.

### Support Vector Machine (SVM)

A support vector machine is a supervised learning method that is usually used for binary classification. It works by fitting a high dimensional function to some training data in order to separate the training data into its two classes. The functional form of this separator can vary, but the simplest example is a hyperplane. Suppose you have a dataset where each entry has `d` dimensions and a binary label. Using a planar kernel, SVM will try to fit a `d` dimensional hyperplane that splits `d`-space so that all observed items on one side of the hyperplane have one label and all observed items on the other side of the hyperplane have the other label. Notice that this does not take likelihood into account, so a standard SVM is a non-probabilistic binary classifier.

In this analysis, SVM is only used for making direction predictions. Take a look at the plot of direction accuracy in `ml/svm.ipynb`. It does not do well. The best stock for predicting direction is Apple with 57% of guesses correct. Why does SVM underperform so badly in this case? Well, SVM has some notable disadvantages. Since it tries to separate the feature space into distinct spaces representing the different classes, it works best when there is good separation between the classes. But consider the stock direction prediction case where the classes are a stock's price going up and down the next day. Our features are things like previous day's closing price, difference between previous day's high and low price, and 7 day average price. Well regardless of what price a stock is (or how it has been trending), it could still go up or down on any given day. While the long term price may be more linked to instrinsic factors of the stock, the day-to-day change is probably more dependent on market news and investor sentiment, which is more fickle. This means the data is not well separated, and therefore makes SVM tough to use in this case.

### Random Forest

Random forest is a very powerful machine learning algorithm. It is an ensemble learning method that works by assembling random decision trees that are trained on random subsets of the training data. While decision trees by themselves have a tendency to greatly overfit training data, a random forest attempts to solve this problem by aggregating over many decision trees (ideally compensating for overfitting with other decision trees that are better fit to other parts of the training data). This enables the random forest to have lower variance as a predictor than a single decision tree. As with KNN, random forest can be used as a classifier for discrete prediction or as a regressor for continuous prediction.

We use both methods in this analysis. Take a look at `ml/random_forest.ipynb` to see how the random forest is used in this analysis and its results. Like the other methods mentioned above, random forest does not do too well. Random forests have two major disadvantages: complexity (may entail many different decision trees) and training time. But when time and complexity are non-factors, random forests are usually quite powerful out-of-the-box ML algorithms. The failure of the random forests in this analysis indicates more problems with our data than with the learning algorithm. It is simply the case that with the very basic stock features used in this analysis, we do not have enough information to accurately predict how the price will move in the next day. Remember, the point of this analysis is to investigate machine learning methods as applied to the stock prediction problem, and not to predict stocks accurately, so please don't use this model to trade.

### Gradient Boosted Regression Trees

Last but not least, this analysis used gradient boosting. This is one of the cleverest and most interesting machine learning methods out there. Like a random forest, it is an ensemble method that is built on many decision trees. However, the decision trees are created sequentially and each is designed to cover for the weaknesses of its predecessors. One can think about this algorithm in the following way: first, train a decision tree on some data. Now take the residuals (the error) from the decision tree prediction to each data point. Train a new decision tree on these residuals. Repeat this process for many iterations and then the final classifier or regressor is just the aggregation of all these decision trees.

Like random forest, gradient boosting is slow to train and computationally expensive. Additionally, because gradient boosting adds weight to its errors, it can weight outliers highly and overfit training data. Still, it is a powerful out-of-the-box machine learning method. In this analysis, however, we see that it, too, does not perform well on the stock data. The bottom line of this analysis is simply that the data used is not sufficient to make accurate stock predictions 
