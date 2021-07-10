import numpy as np
import pandas as pd
from tsmodels import UniformModel, SeasonalModel

class ChainedModels():
  """
  Chain of seasonality which combines different granularity models into one
  chain_of_models: ordered collection of seasonal and uniform models
  which should ascend in frequency
  input_ts: estimate of a low-frequency time series, which the chain converts 
  into high-frequency output time-series in predict
  """
  def __init__(self, chain_of_models, input_ts, calc_func=None):
    if chain_of_models[-1].output_freq != 'h':
      raise Exception('Last model must output hourly frequency')
    self.models = chain_of_models
    self.input_ts = input_ts
    if calc_func is not None:
      for model in self.models:
        if isinstance(model, SeasonalModel):
          model.calculation_func = calc_func
          

  def train(self, training_ts: pd.Series):
    """
    Trains all models in a chain at once using input time series
    training_ts which should have hourly frequency
    """
    if training_ts.index.freq != 'h':
      raise Exception("Training set should have hourly frequency")
    for model in self.models:
      model.train(training_ts.resample(model.output_freq).sum())

  def predict(self, base_ts=None):
    """
    Gets hourly predictions from low-frequency input time-series 
    using a chain of seasonal and uniform models
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
      true_inds = base_ts.resample('h').asfreq().index.values
    else:
      true_inds = self.input_ts.resample('h').asfreq().index.values
    self.prediction = working_ts[true_inds]
    # self.prediction2 = working_ts
    return self.prediction

  def prediction_mae(self, ground_truth, last_periods=None):
    ground_truth = ground_truth[self.prediction.index.values]
    assert all(self.prediction.index == ground_truth.index), \
      "indices of prediction and true timeseries don't align"
    
    self.mae = (ground_truth - self.prediction).abs().mean()
    if last_periods is not None:
      self.mae = \
      (self.prediction - ground_truth).abs()[-last_periods:].mean()
    self.metrics = self.mae
    return self.mae

  def prediction_rmse(self, ground_truth, last_periods=None):
    ground_truth = ground_truth[self.prediction.index.values]
    assert all(self.prediction.index == ground_truth.index), \
      "indices of prediction and true timeseries don't align"
    
    temp_ts = (ground_truth - self.prediction) ** 2
    if last_periods is None:
      last_periods = len(temp_ts)
    self.rmse = temp_ts[-last_periods:].mean() ** 0.5
    self.metrics = self.rmse
    return self.rmse

  def prediction_mqe(self, ground_truth, last_periods=None, q=0.8):
    ground_truth = ground_truth[self.prediction.index.values]
    assert all(self.prediction.index == ground_truth.index), \
      "indices of prediction and true timeseries don't align"
    
    temp_ts = ground_truth - self.prediction
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
  Uses walk-farward cross-validation.
  train_period: integer number of past periods to train model on during each step of 
  cross-validation. If 0, uses all  periods from the start of train_ts 
  ar_period: integer number of past periods values of which to base predictions on.
  If 0, uses all periods from the start of input_ts
  input_ts: time-series with past data, used for cross-val and out of sample predictions
  """
  def __init__(self, chain_of_models, input_ts: pd.Series, train_period: int, 
      ar_period: int, train_ts: pd.Series, step: pd.DateOffset,
      calc_func=None
    ):
    super().__init__(chain_of_models, input_ts, calc_func)
    self.train_period = train_period
    self.ar_period = ar_period  
    self.train_ts = train_ts
    self.step = step

  def description(self):
    """Returns used models, training and prediction parameters"""
    return f"Training period: {self.train_period}, \
    AR (martingale) period: {self.ar_period}, step: {self.step.freqstr} \
    models: {[model.name for model in self.models]}"

  def predict(self, test_period_start: pd.Timestamp, test_period_end: pd.Timestamp):
    """
    For each period after test_period_start it trains seasonal models 
    on data in self.train_ts using train_period last periods
    and gets prediction for that period based on values of ar_period past periods.
    test_period_start: starting date of test period (included)
    test_period_end: date, at and after which no new cycles of testing 
    will be performed
    Returns: predicted high-frequency time-series
    """
    if test_period_start - self.ar_period * self.step < \
       self.input_ts.index[0].start_time:
       raise Exception("ar period is larger than length of input series from \
         its beggining to test period starting point")
    if test_period_start - self.train_period * self.step <\
       self.input_ts.index[0].start_time:
       raise Exception("training period is larger than length of input series from \
         its beggining to test period starting point")

    pred_parts = []
    while test_period_start < test_period_end:
      # Training of the chained model
      if self.train_period == 0:
        train_start = self.train_ts.index[0].start_time
      else:
        train_start = (test_period_start - self.train_period * self.step).\
        strftime('%Y-%m')
      if test_period_start < self.train_ts.index[-1].end_time:
        start_ind = self.train_ts.index.get_loc(test_period_start)
      else: 
        if self.train_period != 0:
          corrected_start = self.train_ts.index[-1].start_time + self.step
          train_start = (corrected_start- self.train_period * self.step).\
          strftime('%Y-%m')
        start_ind = 0
      train_stop  = self.train_ts.index[start_ind-1].end_time.strftime('%Y-%m')
      super().train(self.train_ts[train_start:train_stop])
      # Getting lagged values to base prediction on
      if self.ar_period == 0:
        ar_start = self.input_ts.index[0].start_time
      else:
        ar_start = (test_period_start - self.ar_period * self.step).\
        strftime('%Y-%m')
      if test_period_start < self.input_ts.index[-1].end_time:
        stop_ind = self.input_ts.index.get_loc(test_period_start)
      else: 
        if self.ar_period != 0:
          corrected_start = self.input_ts.index[-1].start_time + self.step
          ar_start = (corrected_start - self.ar_period * self.step).\
          strftime('%Y-%m')
        stop_ind = 0
      ar_stop  = self.input_ts.index[stop_ind-1].end_time.strftime('%Y-%m')
      ar_period = self.input_ts[ar_start:ar_stop]
      # First, we get a base prediction for outermost layer of seasonality model
      # to take as input. It's a simple average of ar_period. Thus, it can be interpreted 
      # either as an ARIMA model (with an underlying hypothesis that input_ts is
      #  at least locally stationary) with its expected value as a next prediction or as 
      # a martingale model (most recent past value(s) is best prediction for the next)
      base_prediction = ar_period.mean()
      ind_val = pd.Period(test_period_start, freq = self.input_ts.index.freq)
      pred_parts.append(
        super().predict(
          base_ts = pd.Series([base_prediction], index=[ind_val])
          )
      )
      test_period_start = test_period_start + self.step
    self.prediction = pd.concat(pred_parts)
    if any(self.prediction.index.duplicated()): 
      raise Exception('Duplicated index', self.train_period, self.ar_period)
    return self.prediction

