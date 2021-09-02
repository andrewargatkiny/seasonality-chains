import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from tsmodels import UniformModel, SeasonalModel

class ChainedModels():
  """
  Chain of seasonality which combines different granularity models into one
  chain_of_models: ordered collection of seasonal and uniform models
  which should ascend in frequency.

  You have to supply the chain with estimated or true low-frequency
  (e. g. monthly or yearly) time-series of interest to get
  high-frequency (e. g. daily or hourly) predictions of its values.

  Attributes
  ----------
  models : list
    Chain of seasonal models which process the time-series.
  input_ts : pd.Series
    An estimate of a low-frequency time series, which the chain converts 
    into high-frequency output time-series in `predict` function.
  calc_func : callable
    A function to use for calculation of seasonality indices inside all models.
    If none, a function specified inside the model is used.
  stype : str
    Seasonality type ('multiplicative' or 'additive')
  """

  def __init__(self, chain_of_models, input_ts, calc_func=None, stype=None):
    self.output_freq =  chain_of_models[-1].output_freq 
    self.models = chain_of_models
    self.input_ts = input_ts
    self.calc_func = calc_func
    self.stype = stype
    if calc_func is not None:
      #print(calc_func)
      for model in self.models:
        if isinstance(model, SeasonalModel):
          model.calculation_func = calc_func
    if stype is not None:
      for model in self.models:
        model.stype = stype
          
  def _model_training(self, model, time_series):
    """Trains a model according to its datatype."""

    resampled = time_series.resample(model.output_freq)
    if model.datatype == 'flow':
      resampled = resampled.sum()
    else:
      resampled = resampled.mean()
    model.train(resampled)

  def train(self, training_ts: pd.Series):
    """
    Trains all models in a chain at once using a single input time 
    series `training_ts`.
    """

    for model in self.models:
      self._model_training(model, training_ts)

  def predict(self, base_ts=None):
    """
    Gets high-frequency predictions from low-frequency input time-series 
    using a chain of models.

    Parameters
    ----------
    base_ts : pd.Series, optional
      When provided, overrides `input_ts` as the source of low-frequncy
      predictions of a time-series.
    """

    if base_ts is not None: working_ts = base_ts.copy()
    else: working_ts = self.input_ts.copy()
    for model in self.models:
      # Biweekly won't work with lower frequences, it'd take a while to program
      # the logic of it
      if model.output_freq == '2w':
        raise Exception('biweekly-output models are not supported')
      working_ts = model.predict(working_ts)

    # Checking if we have more hourly data points at the edges of time-series
    # due to conversion of lower frequences to weeks
    if base_ts is not None: 
      true_inds = base_ts.resample(self.output_freq).asfreq().index.values
    else:
      true_inds = self.input_ts.resample(self.output_freq).asfreq().index.values
    self.prediction = working_ts[true_inds]
    # self.prediction2 = working_ts
    return self.prediction

  def prediction_mae(self, ground_truth, last_periods=None):
    ground_truth = ground_truth.reindex(self.prediction.index)
    assert all(self.prediction.index == ground_truth.index), \
      "indices of prediction and true timeseries don't align"
    
    self.mae = (ground_truth - self.prediction).dropna().abs().mean()
    if last_periods is not None:
      self.mae = \
      (self.prediction - ground_truth).abs()[-last_periods:].mean()
    self.metrics = self.mae
    return self.mae

  def prediction_rmse(self, ground_truth, last_periods=None):
    ground_truth = ground_truth.reindex(self.prediction.index)
    assert all(self.prediction.index == ground_truth.index), \
      "indices of prediction and true timeseries don't align"
    
    temp_ts = (ground_truth - self.prediction).dropna() ** 2
    if last_periods is None:
      last_periods = len(temp_ts)
    self.rmse = temp_ts[-last_periods:].mean() ** 0.5
    self.metrics = self.rmse
    return self.rmse

  def prediction_mqe(self, ground_truth, last_periods=None, q=0.8):
    ground_truth = ground_truth.reindex(self.prediction.index)
    assert all(self.prediction.index == ground_truth.index), \
      "indices of prediction and true timeseries don't align"
    
    temp_ts = (ground_truth - self.prediction).dropna()
    taus = pd.Series([q] * len(temp_ts), index=temp_ts.index)
    taus = taus.where(temp_ts >= 0, q-1)
    temp_ts = temp_ts * taus
    if last_periods is None:
      last_periods = len(temp_ts)
    self.mqe = temp_ts[-last_periods:].mean()
    self.metrics = self.mqe
    return self.mqe
    
class CrossValChain(ChainedModels):
  """
  Gets predictions for future periods using most recent past periods as baseline.
  Uses walk-forward cross-validation.

  Attributes
  ----------
  models : list
    Chain of seasonal models which process the time-series. 
  train_ts: pd.Series
    Time-series used to train all seasonal models. Should have the highest 
    frequency.
  input_ts : int
    time-series with past data, used for cross-val and out of sample predictions.
  step: pd.DateOffset
    A duration of the period for which the chain predicts the time-series
    at each step of cross-val. Should be equal to input_ts' one period length.
  train_period : int
    integer number of past step periods to train model on during each step of 
    cross-validation. If 0, uses all  periods from the start of train_ts.
  ar_period : int
    integer number of past periods values of which to base predictions on.
    If 0, uses all periods from the start of input_ts.
  predict_period : int, default=1
    integer number of step periods to predict at a time, based on 
    previous data, then to move forward this number of steps.
  calc_func : callable
    A function to use for calculation of seasonality indices inside all models.
    If none, a function specified inside the model is used.
  stype : str
    Seasonality type ('multiplicative' or 'additive').
  trend : bool, default=False
    Determines whether to use linear trend for prediction of next step
    frequency period(s).
  deseason : bool, default=False
    Determines whether to deseason last `ar_period` terms of `input_ts`
    before estimation of a low-frequency prediction.
  train_fully : bool, default=False
    Train seasonality weights of all models using full `train_ts`.
  dont_retrain : bool, default=False
    Train seasonality weights of all models only once and don't retrain
    afterwards.
  """

  def __init__(self, chain_of_models, train_ts: pd.Series, input_ts: pd.Series, 
    step: pd.DateOffset, train_period: int, ar_period: int, predict_period=1,
    calc_func=None, stype=None, trend=False, deseason=True, train_fully=False,
    dont_retrain=False
    ):
    super().__init__(chain_of_models, input_ts, calc_func, stype)
    self.train_period = train_period
    self.ar_period = ar_period  
    self.predict_period = predict_period
    self.train_ts = train_ts
    self.step = step
    self.trend = trend
    self.deseason = deseason
    self.train_fully = train_fully
    self.dont_retrain = dont_retrain
    if dont_retrain:
      self.is_trained = False
      if not self.train_fully:
        warnings.warn("You didn't specify you want to train your model on \
          a full dataset, but set dont_retrain as True")

  def description(self):
    """Returns used models, training and prediction parameters"""

    return f"Training period: {self.train_period}, \
    AR (martingale) period: {self.ar_period}, step: {self.step.freqstr}, \
    models: {[model.name for model in self.models]}, \
    trend: {self.trend}, deseason: {self.deseason}"

  def predict(self, test_period_start: pd.Timestamp, test_period_end: pd.Timestamp):
    """
    For each period after test_period_start it trains seasonal models 
    on data in self.train_ts using train_period last periods
    and gets prediction for that period based on values of ar_period past periods.

    Parameters
    ----------
    test_period_start : pd.Timestamp
      Starting date of test/forecast period (included).
    test_period_end : pd.Timestamp
      Date, at and after which no new cycles of testing/forecasting 
    will be performed.

    Returns
    -------
    pd.Series
      Predicted high-frequency time-series.
    """

    if test_period_start - self.ar_period * self.step < \
       self.input_ts.index[0].start_time:
       raise Exception("ar period is larger than length of input series from \
         its beggining to test period starting point")
    if test_period_start - self.train_period * self.step <\
       self.input_ts.index[0].start_time:
       raise Exception("training period is larger than length of input series from \
         its beggining to test period starting point")
    if not self.dont_retrain or not self.is_trained:
      if self.deseason:
        freq = UniformModel._get_frequency_str(self.input_ts)
        if freq == 'w': 
          raise Exception("Deseasoning for weekly periods isn't implemented")
        self.deseason_model = SeasonalModel(\
          freq, datatype=self.models[0].datatype, 
          stype=self.models[0].stype, use_input_freq='y')
        if self.calc_func is not None: 
            self.deseason_model.calculation_func = self.calc_func
      if self.train_fully:
        super().train(self.train_ts)
        if self.deseason:
          self._model_training(self.deseason_model, self.train_ts)
      self.is_trained = True

    pred_parts = []
    while test_period_start < test_period_end:
      # Training of the chained model
      if not self.train_fully and not (self.dont_retrain and self.is_trained):
        self.is_trained = True
        if self.train_period == 0:
          train_start = self.train_ts.index[0].start_time
        else:
          train_start = (test_period_start - self.train_period * self.step).\
          strftime('%Y-%m-%d')
        if test_period_start < self.train_ts.index[-1].end_time:
          ind_train_stop = self.train_ts.index.get_loc(test_period_start)
        else: 
          if self.train_period != 0:
            corrected_start = self.train_ts.index[-1] + self.train_ts.index.freq
            corrected_start = corrected_start.start_time
            train_start = (corrected_start- self.train_period * self.step).\
            strftime('%Y-%m-%d')  
          ind_train_stop = 0
        train_stop  = self.train_ts.index[ind_train_stop-1].end_time.strftime('%Y-%m-%d')
        super().train(self.train_ts[train_start:train_stop])

      # Getting lagged values to base prediction on
      if self.ar_period == 0:
        ar_start = self.input_ts.index[0].start_time
      else:
        ar_start = (test_period_start - self.ar_period * self.step).\
        strftime('%Y-%m-%d')
      if test_period_start < self.input_ts.index[-1].end_time:
        idx_ar_stop = self.input_ts.index.get_loc(test_period_start)
      else: 
        if self.ar_period != 0:
          corrected_start = self.input_ts.index[-1].start_time + self.step
          ar_start = (corrected_start - self.ar_period * self.step).\
          strftime('%Y-%m-%d')
        idx_ar_stop = 0
      ar_stop  = self.input_ts.index[idx_ar_stop-1].end_time.strftime('%Y-%m-%d')
      ar_ts = self.input_ts[ar_start:ar_stop]
      # First, we get a base prediction for outermost layer of seasonality model
      # to take as input. It's a deseasoned average of ar_period. Thus, it can be interpreted 
      # either as an ARIMA model (with an underlying hypothesis that input_ts is
      #  at least locally stationary) with its expected value as a next prediction or as 
      # a martingale model (most recent past value(s) is best prediction for the next)

      # Deseasoning
      if self.deseason:
        if not self.train_fully:
          resampled = self.train_ts[train_start:train_stop]
          self._model_training(self.deseason_model, resampled)
        dummy_ts = ar_ts.copy()
        if self.deseason_model.stype == 'multiplicative':
          dummy_ts[:] = 1
          inds = self.deseason_model.predict(dummy_ts)
          deseasoned_ar = ar_ts / inds
        else:
          dummy_ts[:] = 0
          inds = self.deseason_model.predict(dummy_ts)
          deseasoned_ar = ar_ts - inds
        deseasoned_ar = deseasoned_ar.fillna(0).replace(np.inf, 1)
      else:
        deseasoned_ar = ar_ts
      # High-level prediction for next period(s) via ar or regression
      ind_vals = pd.period_range(test_period_start, periods=self.predict_period,
        freq = self.input_ts.index.freq)
      if not self.trend:
        base_prediction = [deseasoned_ar.mean()] * self.predict_period
      else:
        ind_last = ind_vals[-1]
        ind_first = deseasoned_ar.index[0]
        # We take into account that last period to predict may start long 
        # after the end of ar_ts.
        full_index = pd.period_range(ind_first, ind_last)
        time_axis = np.arange(len(full_index)).reshape(-1, 1)
        reg = LinearRegression().fit(X=time_axis[:len(ar_ts)], y=deseasoned_ar)
        base_prediction = reg.predict(time_axis[-self.predict_period:])
      # Conversion to high frequences
      pred_parts.append(
        super().predict(
          base_ts = pd.Series(base_prediction, index=ind_vals)
          )
      )
      test_period_start = test_period_start + self.step * self.predict_period
    self.prediction = pd.concat(pred_parts)
    if any(self.prediction.index.duplicated()): 
      raise Exception('Duplicated index', self.train_period, self.ar_period)
    return self.prediction

