# seasonality-chains
This is a simple yet powerful framework dedicated to high-frequency forecasting of any time-series which exhibits at least one type of seasonality. It's able to forecast for years ahead with up to one hour granularity. 

The main idea behind the framework and its distinction from others (e. g. statsmodels' seasonal_decompose) is the ability to automatically chain different types of seasonality (daily, weekly, monthly, etc.) into one predictive model which outputs high-frequncy time-series given input of one or several terms of an arbitrary lower frequency. One can freely and flexibly stack them together in one chain as if they are building blocks of a Lego constructor. It's possible to either supply the model with high level manual predictions (e. g. give total value for a next year) or use built-in ARMA/martingale predictions for next periods. 

The framework is a white-box model with simple rationale: real life natural and social processes are often subject not to one, but to many types of seasonality. For example, the number of calls to an emergency service for some particular reason might simultaneously depend on time of the day, day of the week and season (winter, summer) of the year. A chain of seasonality detects and extracts all of these types of seasonality (if any are present) automatically to apply later for a prediction.
![Example of prediction vs actual data](predicted.png)

## Performance in real life scenarios
This framework was in part inspired by and used in Boston Consulting Group (BSG) Gamma hackathon to solve a task of forecasting multiple (~1350) time-series of hourly frequency for a full one quarter of the year, thus yielding $90*24=2160$ data points to predict. It outperforms, both in interpretability and goodness of fit, all other ML models that were tried for achieving best predictions, including regression-based models (incl. Lasso, ElasticNet and SARIMA), ANNs, random forests and boosted trees, and also FaceBook Prophet library, specially dedicated to time-series forecasting.

Examples of the use of the framework, experience with the contest along with optimal solution, and its merits and perfomance in a real life problem are described in an article structured as a Jupyter Notebook and availible at [Contest Solution & Real Life Use](https://nbviewer.jupyter.org/github/andrewargatkiny/seasonality-chains/blob/master/Contest%20Solution%20%26%20Real%20Life%20Use.ipynb). Give it a read!

## Usage, feautures and current limitations
The framework uses 2 types of models which represent building blocks for the chains of seasonality:
* UniformModel(output frequency) – takes input time-series and uniformly upsamples it into output frequency;
* SeasonalModel(output frequency) – can be trained on past high-frequency data to estimate and store seasonal weights (indices) for compatible lower frequnces. It takes input time-series, converts to output frequency using UniformModel and, if its original frequency matches one with availible indices, applies them to the series to get final prediction.

Seasonal model is equipped with three functions which can be used to calculate seasonal indices:
* Mean
* Median
* Quantile (percentile) of an arbitrary order q
The functions calculate generalized values of seasonal periods based on their respective statistics and compare them with overal mean of training time-series to produce either additive or multiplicative type indices.
An output of a model can be an input of another model as long as its frequency is lower or equal than second model's output frequency.

The framework has two built-in types of chains:
* base ChainOfModels which offers flexibility in construction and selection of data for training the models and forming predictions on;
* CrossValChain which trains seasonalal models and bases predictions on recent data of user-specified time-windows (train_perdiod and ar_period). It employs simple MA model to predict next high-level values of input, which is conceptually close to ARMA or martingale random processes models, depending on number of considered past lags. Given a past time-series of sufficient length, it can perform a one-step walk-forward time-series validation, hence the name CrossValChain.

In-depth examples of the use of these types of chains are in the Jupyter Notebook file.

This library is a work in progress. Current limitations:
* When in multiplicative mode, works correctly only with nonnegative input time-series.
* Predictions with frequencies higher than one hour are not tested
* So final output is always of hourly frequency
* ARMA calculation is a simple moving average of ar_period preceeding periods (no prior deseasoning supported). If ar_period=1, it’s a martingale model.
* No trend models, only stationary ARMA/martingale based predictions

Later I'll wrap up a detailed article documenting the process of the framework's development, its rationale and concepts used, and a guide for its features, so stay tuned. If you have any comments, suggestions of advice, please contact me. Also, I'd greatly appreciate directions on which datasets can be used to test the framework with.
