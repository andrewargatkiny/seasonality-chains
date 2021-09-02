from calendar import weekday
import numpy as np
import pandas as pd

mappings = {
    'h': [('d', 24), ('w', 7*24), ('y', 24*365)],
    '2h': [('d', 12), ('w', 7*12), ('y', 12*365)],
    '4h': [('d', 6), ('w', 7*6), ('y', 6*365)], 
    '8h': [('d', 3), ('w', 7*3), ('y', 3*365)], 
    'd': [('w', 7), ('y', 365)],
    'm': [('q', 3), ('y', 12)],
    'q': [('y', 4)]
}
class TimeModel():
    
    def __init__(self, output_freq, stype='multiplicative'):
      self.output_freq = output_freq
      self.stype = stype

class MedianModel(TimeModel):
    """Produces output time-series all values of which are median of 
    previous period of some length"""

    def __init__(self, output_freq, stype='multiplicative', datatype='flow'):
      """
      Initialises a model with given output frequency
      """

      super().__init__(output_freq, stype=stype)
      self.name = 'MedianModel ' + self.output_freq
      self.datatype = datatype

    def train(self, time_series: pd.Series):
      """Trains the model by calculating median of input time-series"""

      if time_series.index.freq != self.output_freq:
        raise Exception('Input time-series must have the same frequency as\
        desired output frequency')   
      self.median = time_series.median()

    def predict(self, time_series):
      """
      (Optionally) converts low-frequency time_series into desired higher 
      frequency, and replaces all values in it by a single value - 
      median calculated previously in train()
      """

      if time_series.isnull().any(): 
        raise Exception(time_series, 'None values in input time series')
      inds = time_series.resample(self.output_freq).asfreq().index
      prediction = pd.Series([self.median] * len(inds), index=inds)
      return prediction


class UniformModel(TimeModel):
    """
    Produces output time-series, uniformly partitioning input 
    time series values to get desired frequency.
    Attributes
    ----------
    output_freq : str
      Desired frequency of output time-series. Should be compatible
      with Pandas frequency names (e.g. 'd', 'w', 'y').
    stype : str, defaut='multiplicative'
      Seasonality type ('multiplicative' or 'additive').
    datatype : str, default='flow'
      Type of time-series data assumed. Can take two values: 
      'flow' and 'stock'.
    name : str
      Description of the model.
    """

    def __init__(self, output_freq, stype='multiplicative', datatype='flow'):
      """Initialises a model with given output frequency"""

      super().__init__(output_freq, stype=stype)
      self.name = 'UniformModel ' + self.output_freq
      self.datatype = datatype

    def train(self, *args):
      """Doesn't really train the model. Used for compatibility reasons"""

      pass

    def predict_1step(self, total_value, input_freq, start_dt):
        """
        Divides one term of low-frequency time-series into several terms with higher
        frequency (e. g. 1d -> 24h).

        Parameters
        ----------
        total_value : float
          Value of low frequency term (predicted total of high-frequency.
        terms) 
        input_freq : str
          Length (and frequency) of the period during which the value.
        would occur    
        start_dt : pd.Timestamp
          The beggining of this period.
        
        Returns
        -------
        pd.Series
          Output high-frequency series.
        """
        # if start_dt.hour != 0:
        #     print(start_dt)
        #     raise Exception("your sample's first date must begin from 00:00")

        input_period = pd.Period(start_dt, input_freq)
        end_dt = input_period.end_time

        # If length of input period is less than frequency of resulting one 
        # (e.g. 1D < 1W)
        if end_dt < pd.Period(start_dt, self.output_freq).end_time:
            print(end_dt, start_dt, self.output_freq)
            raise Exception('input frequency < output frequency')
        
        # Time-series with lower frequencies should always start on a
        #  1st day of a month
        if input_freq in ['m', '2m', '3m', 'q', 'y'] and \
        not start_dt.is_month_start:
            raise Exception(f"your sample's first date must be first date of \
                            a month if you use {input_freq} frequency")
        
        out_start_dt = start_dt
        out_end_dt = end_dt

        # If output_freq is a multiple of week, then unfitting input_period
        # gets complemented to a multilple of week with an adjustment made
        # for total_value
        if self.output_freq == 'w': # in ['w', '2w']:
            unaligned_week_lday = False
            last_day_shift = 0
            # Nasty workoaround but seems bug-proof
            old_elapsed_days = (end_dt - start_dt).round('d').days
            if start_dt.weekday() != 0:
                out_start_dt = start_dt.to_period('w').start_time
            if end_dt.weekday() != 6:
                out_end_dt = end_dt.to_period('w').end_time
                unaligned_week_lday = True
                last_day_shift = end_dt.weekday() + 1
            if self.datatype == 'flow':
              new_elapsed_days = (out_end_dt - out_start_dt).round('d').days
              total_value *= new_elapsed_days / old_elapsed_days

        output_index = pd.period_range(
            start = out_start_dt.to_period(self.output_freq),
            end = out_end_dt.to_period(self.output_freq),
            freq = self.output_freq
        )
        # Just dividing total_value by number of new periods
        if total_value != 0:
          if self.datatype == 'flow':
            pred_vals = total_value / len(output_index)
          else:
             pred_vals = total_value
        else:
          pred_vals = 0
        result_ts = pd.Series(pred_vals, index=output_index)
        
        if self.output_freq == 'w':
          return (result_ts, unaligned_week_lday, last_day_shift)
        else:
          return result_ts
    def _get_frequency_str(ts: pd.Series):
      """
      Returns
      -------
      str
        String representation of a series' frequency in terms
      used by the framework.
      """

      # Ugly Hack
      input_freq = ts.index.freq.freqstr.lower().split('-')[0]
      if input_freq == 'a': input_freq = 'y'
      return input_freq

    def predict(self, time_series):
      """
      Converts low-frequency time_series into higher frequency, applying a model
      of frequency upsampling to each term of the series separately and combining
      and combining results into a single sequence.

      Returns
      -------
      pd.Series
        Output high-frequency series.
      """

      self.input_freq = UniformModel._get_frequency_str(time_series)
      
      if time_series.isnull().any(): 
        raise Exception(time_series, 'None values in input time series')
      
      if self.input_freq == self.output_freq:
        return time_series.copy()
      # A shortcut to speed up the calculations
      if self.output_freq != 'w':
        if self.datatype == 'flow':
          prediction = time_series.resample(self.output_freq).asfreq() \
          .fillna(0).resample(self.input_freq).transform(np.mean)
        else:
          prediction = time_series.resample(self.output_freq).asfreq() \
          .fillna(method='pad')
        return prediction

      parts = []
      # Getting high-level prediction lists
      for i in range(len(time_series)):
        start_time = time_series.index[i].start_time
        parts.append(
            self.predict_1step(time_series[i], self.input_freq, start_time))
        
      # If we need to get weekly data, we combine together 2 predictions for
      # one last/first week in the junction of 2 series for different periods
      if self.output_freq == 'w':
        for i in range(len(parts) - 1):
          curr_part = parts[i][0]
          next_part = parts[i+1][0]
          indic_shift = parts[i][1]
          shift = parts[i][2]
          # Forecast for a junction week is a weighted average of two forecasts
          if (indic_shift):
            parts[i][0].iloc[-1] = \
            curr_part.iloc[-1] * shift/7 + next_part.iloc[0] * (1-shift/7)
            parts[i+1][0].drop(index= next_part.index[0], inplace=True)
        for i in range(len(parts)):
          parts[i] = parts[i][0]

      prediction = pd.concat(parts)

      if prediction.isnull().any(): 
        raise Exception('None values in output series')
      return prediction

class SeasonalModel(UniformModel):
    """
    A model which predicts higher frequency output time-series 
    applying seasonality ratios to input lower frequency time-series.

    Attributes
    ----------
    output_freq : str
      Desired frequency of output time-series. Should be compatible
      with Pandas frequency names (e.g. 'd', 'w', 'y').
    stype : str, defaut='multiplicative'
      Seasonality type ('multiplicative' or 'additive').
    calculation_func : callable, default=get_seasonal_weights
      One of the class functions for calculation of seasonality indices.
    datatype : str, default='flow'
      Type of time-series data assumed. Can take two values: 
      'flow' and 'stock'.
    use_input_freq : str, optional
      If not None, uses specified frequency in `predict` method instead 
      of input time-series one by applying seasonal weights of this 
      frequency to output time-series.
    name : str
      Description of the model.
    """

    def check_dates_align(self, time_series, freq_name, 
    drop_leap_day=True):
      """
      If time_series doesn't begin at first moment of specified
      freq_name, the function returns ts with indices complemented 
      to the start and none values, else returns input ts.
      It also optionally removes all leap day values from time_series.
      """

      def is_period_start(freq: str, time: pd.Timestamp):
        if freq == 'y': 
          return time.is_year_start
        if freq == 'q': 
          return time.is_quarter_start
        if freq == 'w': 
          return time.dayofweek == 0
        if freq == 'h': 
          return time.hour == 0
      
      for freq in ['y', 'q', 'w', 'd']:
        if freq_name == freq and \
        not is_period_start(freq, time_series.index[0].start_time):
          start = time_series.index[0].start_time.to_period(freq).start_time
          end = pd.Timestamp(time_series.index[-1].start_time)
          index = pd.period_range(start, end, freq=self.output_freq)
          time_series = time_series.reindex(index)
          break

      # When calculating hourly or daily seasonality by year, it's 
      # necessary to delete leap day
      if drop_leap_day and self.output_freq in ['h', '2h', '4h', '8h', 'd'] \
                                    and freq_name == 'y':
        inds_to_drop = time_series.index[(time_series.index.month == 2) 
                                & (time_series.index.day == 29)].values
        time_series = time_series.drop(inds_to_drop)
      return time_series
      
        
    def get_seasonal_weights(self, ts, period):
      """
      Calculates seasonality using mean values of periodic fluctuations compared 
      to overall mean.

      Parameters
      ----------
      ts : pd.Series
        Input time-series to derive the weights from.
      period: int
        The period of seasonality in terms of `ts` frequency

      Returns
      -------
      pd.Series
        Seasonal weights indexed from 0 to `period`-1
      """

      ts = ts.copy()
      #print(int(len(ts) / period), len(ts), period)
      new_level = np.tile(np.arange(period), int(np.ceil(len(ts) / period)))
      ts.index = new_level[:len(ts)] 
      ts = ts.dropna()
      #print(ts)
      seasons = ts.groupby(level=0).mean().rename('periods').to_frame()
      seasons['mean'] = seasons['periods'].mean()
      #print(seasons, '\n________')
      if self.stype == 'multiplicative':
        return (seasons['periods'] / seasons['mean']).fillna(1)
      else:
        return (seasons['periods'] - seasons['mean']).fillna(0)
      
    def get_quantile_weights(self, ts, period, q=0.8):
      """
      Calculates seasonality using quantile values of periodic fluctuations 
      compared to overall mean of ts.

      Parameters
      ----------
      ts : pd.Series
        Input time-series to derive the weights from.
      period: int
        The period of seasonality in terms of `ts` frequency
      q : float, default=0.8
        quantile to use

      Returns
      -------
      pd.Series
        Seasonal weights indexed from 0 to `period`-1
      """
      ts = ts.copy()
      new_level = np.tile(np.arange(period), int(np.ceil(len(ts) / period)))
      ts.index = new_level[:len(ts)] 
      ts = ts.dropna()
      seasons = ts.groupby(level=0).quantile(q).rename('periods').to_frame()
      #seasons['mean'] = seasons['periods'].mean()
      seasons['mean'] = ts.mean()
      if self.stype == 'multiplicative':
        weights = (seasons['periods'] / seasons['mean']).fillna(1).\
        replace(np.inf, 1)
      else:
        weights = (seasons['periods'] - seasons['mean']).fillna(0).\
        replace(np.inf, 0)
      return weights

    def get_median_weights(self, ts, period):
      """
      Calculates seasonality using median values of periodic fluctuations 
      compared to overall mean of ts

      Parameters
      ----------
      ts : pd.Series
        Input time-series to derive the weights from.
      period: int
        The period of seasonality in terms of `ts` frequency

      Returns
      -------
      pd.Series
        Seasonal weights indexed from 0 to `period`-1
      """

      ts = ts.copy()
      new_level = np.tile(np.arange(period), int(np.ceil(len(ts) / period)))
      ts.index = new_level[:len(ts)] 
      ts = ts.dropna()
      seasons = ts.groupby(level=0).median().rename('periods').to_frame()
      #seasons['mean'] = seasons['periods'].mean()
      seasons['mean'] = ts.mean()
      if self.stype == 'multiplicative':
        weights = (seasons['periods'] / seasons['mean']).fillna(1).\
        replace(np.inf, 1)
      else:
        weights = (seasons['periods'] - seasons['mean']).fillna(0).\
        replace(np.inf, 0)
      return weights
    
    def get_experimental_weights2(self, ts, period):
      alpha=0.01
      ts = ts.copy()
      new_level = np.tile(np.arange(period), int(np.ceil(len(ts) / period)))
      ts.index = new_level[:len(ts)] 
      ts = ts.dropna()
      seasons = ts.groupby(level=0).mean().rename('periods').to_frame()
      seasons['mean'] = seasons['periods'].mean()
      seasons['low_quantile'] = ts.groupby(level=0).quantile(alpha)
      seasons['high_quantile'] = ts.groupby(level=0).quantile(1-alpha)
      seasons['seasonality'] = seasons['high_quantile'].where(
          seasons['high_quantile'] - seasons['mean'] >= 
          seasons['mean'] - seasons['low_quantile'],
          seasons['low_quantile']
      )
      if self.stype == 'multiplicative':
        return (seasons['seasonality'] / seasons['mean']).fillna(1)
      else:
        return (seasons['seasonality'] - seasons['mean']).fillna(0)

    def __init__(self, output_freq, stype='multiplicative', 
        calc_func=get_seasonal_weights, datatype='flow',
        use_input_freq=None):
      """
      Initialises a model with given output frequency
      """

      super().__init__(output_freq, stype=stype, datatype=datatype)
      self.calculation_func = calc_func
      self.name = 'SeasonalModel ' + self.output_freq
      if use_input_freq is not None: 
        self.name = self.name + " from " + use_input_freq
      self.use_input_freq = use_input_freq
      
    def train(self, time_series, verbose=False):
      """
      This function calculates seasonality indices for all possible input 
      frequences suitable for output_freq, and saves them to self.seasonal_weights
      """

      if time_series.index.freq != self.output_freq:
          raise Exception('training dataset must have same \
          frequency as output series')
      base_ts = time_series
      self.seasonal_weights = dict()
      # For each useful input frequency for the output frequency
      for freq_name, freq in mappings[self.output_freq]:
        curr_ts = base_ts.copy()
        # We get aligned data, drop leap day indices and then calculate seasonality
        curr_ts = self.check_dates_align(curr_ts, freq_name)
        
        # If number of periods for one cycle of seasonality is greater than
        # number of periods in input time series
        if freq > len(time_series):
          self.seasonal_weights.update({
            freq_name : pd.Series([1] * freq, index=np.arange(freq))
          }) 
          continue
        """        
        # If we look at weekly seasonality, it's necessary to ensure each week 
        # starts on Monday and ends on Sunday
        if freq_name == 'w':
          while(curr_ts.index[0].dayofweek != 0):
            curr_ts = curr_ts.drop(curr_ts.index[0])
          while(curr_ts.index[-1].dayofweek != 6):
            curr_ts = curr_ts.drop(curr_ts.index[-1])
        """
        if verbose:
          print(freq_name, freq)  
          print(curr_ts)    
        self.seasonal_weights.update({
            freq_name : self.calculation_func(self, curr_ts, freq)
        }) 
        
    def predict(self, time_series):
      """
      Converts lower-frequency time_series into higher-frequency, applying a model of
      frequency magnification separately to each term of the series and combining 
      results into a single sequence.

      Returns
      -------
      pd.Series
        Predicted high-frequency time-series.
      """ 

      prediction = UniformModel.predict(self, time_series)
      prediction.rename('prediction', inplace=True)
      input_freq = self.input_freq
      if (self.output_freq in ['m', 'q']) and (input_freq in ['m', 'q']):
        input_freq = 'y'
      if self.use_input_freq:
        input_freq = self.use_input_freq
      if input_freq in [item[0] for item in mappings[self.output_freq]]:
        aligned_index = self.check_dates_align(prediction, input_freq).\
            index
        l = len(self.seasonal_weights[input_freq])
        inds_np = np.tile(
            self.seasonal_weights[input_freq].index.values,
            int(np.ceil(len(aligned_index)/l))
        )
        new_inds = pd.Series(
          inds_np[:len(aligned_index)], index = aligned_index).rename('aligned')
        pred_w_inds = pd.concat([prediction, new_inds], axis = 1)
        seasonality = self.seasonal_weights[input_freq].rename('seasonality')
        
        # Dealing with leap days.
        if self.output_freq in ['h', '2h', '4h', '8h', 'd'] \
          and input_freq == 'y' and pred_w_inds.index.is_leap_year.any():
          inds_to_replace = pred_w_inds.index[(pred_w_inds.index.month == 2) 
                                  & (pred_w_inds.index.day == 29)].values
          if len(inds_to_replace) > 0:
            years = pred_w_inds.loc[inds_to_replace].index.year.unique()
            leap_per_length = int(len(inds_to_replace) / len(years))
            replacing_vals = pred_w_inds.shift(leap_per_length).loc[
              inds_to_replace, 'prediction']
            pred_w_inds.loc[inds_to_replace, 'prediction'] = replacing_vals 
            # Negative index to ensure there's no collisions
            leap_day_concat_inds = np.array((range(0, -leap_per_length, -1))) -1
            pred_w_inds.loc[inds_to_replace, 'aligned'] = \
              leap_day_concat_inds.tolist() * len(years)
            # not 59 because of 0 based indexing
            prev_day_to_leap = 58 * leap_per_length
            seasonality = seasonality.append(pd.Series(
              seasonality.iloc[
                prev_day_to_leap: prev_day_to_leap + leap_per_length].values,
              index = leap_day_concat_inds
            )).rename('seasonality')

        pred_w_inds = pred_w_inds.dropna()
        pred_w_seasons = pd.merge(left=pred_w_inds, right=seasonality,
            left_on='aligned', right_index=True, how='left'
        )
        if self.stype=='multiplicative':
            pred_w_seasons['seasonality'].fillna(1, inplace=True)
        else:
            pred_w_seasons['seasonality'].fillna(0, inplace=True)

        if self.stype=='multiplicative':
            prediction = pred_w_seasons['prediction'] * \
                pred_w_seasons['seasonality']
        else:
            prediction = pred_w_seasons['prediction'] + \
                pred_w_seasons['seasonality']                   
      return prediction

class TimeWeekOfYear(SeasonalModel):
    """Predicts output time-series with frequences higher or equal to
    daily simultaneously taking into accont two types of seasonality:
    a moment during the week and its approximate location within the year
    timespan.
    
    See also: `Seasonal Model`.
    Attributes:
    """

    HOLIDAYS = [
        'January 1', 'January 2', 'January 3', 'January 4', 'January 7',
        'February 23', 'March 8', 'May 1', 'May 9', 'June 12', 'November 4', 
        'December 29', 'December 30', 'December 31'
    ]
    ROLLING_HOLIDAYS = {
      'Easter': {'2019': 'April 28', '2020': 'April 19', '2021' : 'September 2'},
      'Unnamed': {'2019': 'October 25', '2020': 'October 26', '2021' : 'October 27'}
    }
    def __init__(self, output_freq, stype='multiplicative', datatype='flow',
    calc_func=SeasonalModel.get_seasonal_weights,
    excl_holidays=True, before_hol=0, after_hol=0):
      """
      Initialises a model with given output frequency.

      Parameters
      ----------
      excl_holidays : bool, default=True
        Calculate seasonality indices for user-defined holidays and 
        rolling holidays by exact yearly seasonality with no use of 
        weekday alignment.
      before_hol : int, default=0
        Calculate also this number of days before holidays or rolling
        holidays by exact yearly seasonality.
      after_hol : int, default=0
        Calculate also this number of days after holidays or rolling
        holidays by exact yearly seasonality.
      """

      if output_freq not in ['h', '2h', '4h', '8h', 'd']:
        raise Exception('TimeWeekOfYear model is able to output only up\
          to a daily frequency')
      super().__init__(output_freq, stype=stype, calc_func=calc_func)
      self.name = ' '.join(['TimeWeekOfYear', self.output_freq,
      'excl_holidays:', str(excl_holidays), 'before_hol:', str(before_hol),
      'after_hol:', str(after_hol)]) 
      self.datatype = datatype
      self.use_input_freq = 'y'
      self.excl_holidays = excl_holidays
      self.before_hol = before_hol
      self.after_hol = after_hol

    def _get_holiday_dates(self, year):
      """Gets daily indices of holidays for current year"""

      dates = []
      for hol_date in TimeWeekOfYear.HOLIDAYS:
        curr_date = pd.Timestamp(hol_date + ' ' + year, freq='d')
        for rel_time in range(self.before_hol + 1):
          dates.append(curr_date - rel_time * pd.DateOffset())
        dates.append(curr_date)
        for rel_time in range(self.after_hol + 1):
          dates.append(curr_date + rel_time * pd.DateOffset())
      # Converting them to higher frequency if necessary
      dates_per = pd.PeriodIndex(dates, freq='d')
      if self.output_freq == 'd':
        dates = dates_per
      else:
        dates = pd.Series(0, index=dates_per).resample(self.output_freq)\
        .asfreq().index
      return dates
    def _get_roll_holiday_dates(self, year, hol_name):
      """Gets daily indices of a rolling holiday for current year"""

      dates = []
      if year in TimeWeekOfYear.ROLLING_HOLIDAYS[hol_name]:
        hol_date = TimeWeekOfYear.ROLLING_HOLIDAYS[hol_name][year]
        curr_date = pd.Timestamp(hol_date + ' ' + year, freq='d')
      for rel_time in range(self.before_hol + 1):
        dates.append(curr_date - rel_time * pd.DateOffset())
      dates.append(curr_date)
      for rel_time in range(self.after_hol + 1):
        dates.append(curr_date + rel_time * pd.DateOffset())
      # Converting them to higher frequency if necessary
      dates_per = pd.PeriodIndex(dates, freq='d').unique().sort_values()
      if self.output_freq == 'd':
        dates = dates_per
      else:
        dates = pd.Series(0, index=dates_per).resample(self.output_freq)\
        .asfreq().index
      return dates

    def train(self, time_series: pd.Series):
      """Trains seasonality indices shifting values ofeach year of data 
      so that their first days of year fall on the same weekday"""

      if time_series.index.freq != self.output_freq:
        raise Exception('training dataset must have same \
        frequency as output series')
      # Need to refactor later
      self.multiplier = 1
      if self.output_freq == 'h': self.multiplier = 24
      elif self.output_freq == '2h': self.multiplier = 48
      # Getting weekdays of first day in all years of time_series
      years = time_series.index.year.unique().astype(str)
      inds_first_day = time_series.index.to_timestamp().is_year_start
      first_days = time_series[inds_first_day] \
        .index.weekday[::self.multiplier]
      if len(first_days) != len(years):
        raise Exception('You must supply a full year worth of data, for\
          at least one year')
      self.first_weekday = first_days[0]
      # Creating storage for raw rolling holidays indices
      if self.excl_holidays:
        roll_hols_seasonals = {}
        for roll_hol in TimeWeekOfYear.ROLLING_HOLIDAYS:
          roll_hols_seasonals[roll_hol] = []
      # Shifting each year's data so every year starts on the same
      # day of week.
      #yearly_ts = [time_series[years[0]]]
      yearly_ts = []
      temp_ts = time_series.copy()
      for i in range(len(first_days)):
        # Dealing with rolling holidays
        if self.excl_holidays:
          for roll_hol in TimeWeekOfYear.ROLLING_HOLIDAYS:
            dates = self._get_roll_holiday_dates(years[i], roll_hol)
            hol_coefs = (temp_ts[dates] / temp_ts[years[i]].mean())\
              .reset_index(drop=True)
            roll_hols_seasonals[roll_hol].append(hol_coefs)
            # Replacing values at a rolling holiday date by a week prior ones
            temp_ts[dates] = temp_ts.shift(7 * self.multiplier)[dates]
        shift = (first_days[i] - self.first_weekday) % 7
        if shift > 3: shift = shift - 7
        temp_ts = temp_ts.shift(shift * self.multiplier)[years[i]]
        yearly_ts.append(temp_ts)
      for ts in yearly_ts:
        if ts.index[0].start_time.is_leap_year:
          ts = ts[:-self.multiplier]
          #print(ts)
      # Training on weekly aligned time series
      aligned_ts = pd.concat(yearly_ts)
      self.weekmoment_weights = self.calculation_func(
        self, aligned_ts, self.multiplier * 365
      )
      # Obtaining final seasonal weights for rolling holidays
      # Not optimized for median and quantile weights
      if self.excl_holidays:
        self.r_hol_seas_weights = {}
        for roll_hol in TimeWeekOfYear.ROLLING_HOLIDAYS:
          mean_indices = sum(roll_hols_seasonals[roll_hol]) \
            / len(roll_hols_seasonals[roll_hol])
          self.r_hol_seas_weights[roll_hol] = mean_indices
      # Training usual seasonal model to have indices for regular holidays
      # and missing values in shifted model.
      super().train(time_series)
      self.standard_weights = self.seasonal_weights['y']

    def predict(self, time_series):
      """
      Converts lower-frequency time_series into higher-frequency, applying a model of
      frequency magnification separately to each term of the series and combining 
      results into a single sequence 
      """

      prediction = UniformModel.predict(self, time_series)
      prediction.rename('prediction', inplace=True)
      input_freq = 'y'
      #.to_period(freq).start_time
      years = prediction.index.year.unique().astype(str)
      inds_first_day = prediction.index.to_timestamp().is_year_start
      first_days = prediction[inds_first_day].index.weekday[::self.multiplier]
      if len(first_days) != len(years):
        first_days = [prediction[year].index[0].start_time.\
        to_period('y').start_time.weekday() for year in years]
      # For each year in uniformly predicted time-series adjusting it with
      # seasonal predictions aligned to fit same weekdays
      weekmoment_preds = []
      for i in range(len(years)):
        curr_prediction = prediction[years[i]]
        curr_first_weekday = first_days[i]
        shift = (self.first_weekday - curr_first_weekday) % 7
        if shift > 3: shift = shift - 7
        #print(curr_first_weekday, self.first_weekday, shift)
        # Replacing NAs caused by shifting with yearly seasonality
        curr_seasonality = self.weekmoment_weights.shift(shift * 
          self.multiplier)
        mask = curr_seasonality.isna()
        curr_seasonality[mask] = self.standard_weights[mask]
        # Aligning indices and placing them together
        aligned_index = self.check_dates_align(curr_prediction, input_freq, 
          drop_leap_day=False).index
        l = len(self.weekmoment_weights)
        inds_np = np.tile(
            self.weekmoment_weights.index.values,
            int(np.ceil(len(aligned_index)/l))
        )
        new_inds = pd.Series(
          inds_np[:len(aligned_index)], index = aligned_index).rename('aligned')
        pred_w_inds = pd.concat([curr_prediction, new_inds], axis = 1)
        pred_w_inds = pred_w_inds.dropna()
        # Getting primary week-of-the-year prediction for current year
        seasonality = curr_seasonality.rename('seas_weekmom')
        pred_w_seasons = pd.merge(left=pred_w_inds, right=seasonality,
            left_on='aligned', right_index=True, how='left'
        )
        if self.stype=='multiplicative':
            pred_w_seasons['weekmom_pred'] = pred_w_seasons['prediction'] * \
                pred_w_seasons['seas_weekmom']
        else:
            pred_w_seasons['weekmom_pred'] = pred_w_seasons['prediction'] + \
                pred_w_seasons['seas_weekmom']                   
        # Replacing NAs which happen at the end of any leap year and
        # holidays with just a yearly seasonality
        nulls = pred_w_seasons['weekmom_pred'].isnull()
        if nulls.any() or self.excl_holidays:
          pred_w_seasons['yearly_pred'] = super().predict(curr_prediction)
        if nulls.any():
          inds_null = pred_w_seasons[nulls]
          pred_w_seasons.loc[inds_null, 'weekmom_pred'] = \
            pred_w_seasons.loc[inds_null, 'yearly_pred']
        if self.excl_holidays:
          # Getting daily indices of holidays for current year
          dates = self._get_holiday_dates(years[i])
          dates = dates.intersection(pred_w_seasons.index)
          # Replacing weekday predictions of holidays and their preceding
          # and following dates
          pred_w_seasons.loc[dates, 'weekmom_pred'] = \
            pred_w_seasons.loc[dates, 'yearly_pred']
          # Getting values for rolling holidays
          for roll_hol in TimeWeekOfYear.ROLLING_HOLIDAYS:
            dates = self._get_roll_holiday_dates(years[i], roll_hol)
            curr_roll_hol = self.r_hol_seas_weights[roll_hol]. \
              reset_index(drop=True)
            curr_roll_hol.index = dates
            dates = dates.intersection(pred_w_seasons.index)
            # Base prediction is uniform so index 0 will do fine
            base_value = pred_w_seasons['prediction'][0]
            if self.stype=='multiplicative':
              final_hol_vals = base_value * curr_roll_hol[dates]
            else:
              final_hol_vals = base_value + curr_roll_hol[dates]
            pred_w_seasons.loc[dates, 'weekmom_pred'] = final_hol_vals
          
        weekmoment_preds.append(pred_w_seasons['weekmom_pred'])
        #print(pred_w_seasons[90:].head(60))

      prediction = pd.concat(weekmoment_preds)
      return prediction
