# seasonality-chains
This is a simple yet powerful framework dedicated to high-frequency forecasting of any time-series which exhibits at least one type of seasonality. It's able to forecast for years ahead with up to one hour granularity. 

The main idea behind the framework and its distinction from others (e. g. statsmodels' seasonal_decompose) is the ability to automatically chain different types of seasonality (daily, weekly, monthly, etc.) into one predictive model which outputs high-frequncy time-series given input of one or several terms of an arbitrary lower frequency. One can freely and flexibly stack them together in one chain as if they are building blocks of a Lego constructor. It's possible to either supply the model with high level manual predictions (e. g. give total value for a next year) or use built-in ARMA/martingale predictions. 

This framework was in pat inspired by and used in Boston Consulting Group (BSG) Gamma hackathon to solve a task of forecasting multiple (~1350) time-series of hourly frequency for one quarter of the year, thus yielding $90*24=2160$ data points to predict. It outperforms, both in interpretability and goodness of fit, all other ML models that were tried for achieving best predictions, including regression-based models (incl. Lasso, ElasticNet and SARIMA), ANNs, random forests and boosted trees, and also FaceBook Prophet library, specially dedicated to time-series forecasting.

Examples of the use of the framework, experience with the contest along with optimal solution, and its merits and perfomance in a real life problem are described in an article structured as a Jupytern Notebook and availible at https://github.com/andrewargatkiny/seasonality-chains/blob/master/Contest%20Solution%20%26%20Real%20Life%20Use.ipynb. Give it a read!

The framework uses 2 types of models which represent building blocks for the chains of seasonality:
* UniformModel(<output frequency>) – takes input time-series and uniformly upsamples it into output frequency;
* SeasonalModel(<output frequency>) – can be trained to store seasonal weights (indices) for compatible lower frequnces. It takes input time-series, converts to output frequency using UniformModel and, if its original frequency matches one with availible indices, applies them to the series to get final prediction.
