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

class UniformModel(TimeModel):
    """Produces output time-series, uniformly partitioning input 
    time series values to get desired frequency"""

    def __init__(self, output_freq, stype='multiplicative'):
      """
      Initialises a model with given output frequency
      """
      super().__init__(output_freq, stype='multiplicative')
      self.name = 'UniformModel ' + self.output_freq

    def train(self, *args):
      pass   

    def predict_1step(self, total_value, input_freq, start_dt):
        """
        Divides one term of low-frequency time-series into several terms with higher
        frequency (e. g. 1d -> 24h)
        total_value: value of low frequency term (predicted total of high-frequency 
        terms) 
        input_freq: length (and frequency) of the period during which the value
        would occur    
        start_dt: the beggining of this period
        """
        """
        if start_dt.hour != 0:
            print(start_dt)
            raise Exception("your sample's first date must begin from 00:00")
        """    
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

            new_elapsed_days = (out_end_dt - out_start_dt).round('d').days
            total_value *= new_elapsed_days / old_elapsed_days

        output_index = pd.period_range(
            start = out_start_dt.to_period(self.output_freq),
            end = out_end_dt.to_period(self.output_freq),
            freq = self.output_freq
        )
        # Just dividing total_value by number of new periods
        if total_value != 0:
          pred_vals = total_value / len(output_index)
        else:
          pred_vals = 0
        result_ts = pd.Series(pred_vals, index=output_index)
        
        if self.output_freq == 'w':
          return (result_ts, unaligned_week_lday, last_day_shift)
        else:
          return result_ts

    def predict(self, time_series):
      """
      Converts low-frequency time_series into higher frequency, applying a model
      of frequency upsampling to each term of the series separately and combining
      and combining results into a single sequence.
      """
      # Ugly Hack
      input_freq = time_series.index.freq.freqstr.lower().split('-')[0]
      if input_freq == 'a': input_freq = 'y'
      self.input_freq = input_freq
      
      if time_series.isnull().any(): 
        raise Exception(time_series, 'None values in input time series')
      
      # A shortcut to speed up the calculations
      if self.output_freq != 'w':
        prediction = time_series.resample(self.output_freq).asfreq() \
        .fillna(0).resample(input_freq).transform(np.mean)
        return prediction

      parts = []
      # Getting high-level prediction lists
      for i in range(len(time_series)):
        start_time = time_series.index[i].start_time
        parts.append(
            self.predict_1step(time_series[i], input_freq, start_time))
        
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
    applying seasonality ratios to input lower frequency time-series
    """
    def check_dates_align(self, time_series, freq_name):
      """
      If time_series doesn't begin at first moment of specified
      freq_name, the function returns ts with indices complemented 
      to the start and none values, else returns input ts.
      It also removes all leap day values from time_series.
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

      # When calculating hourly or daily seasolanity by year, it's 
      # necessary to delete leap day
      if self.output_freq in ['h', '2h', '4h', '8h', 'd'] \
                                    and freq_name == 'y':
        inds_to_drop = time_series.index[(time_series.index.month == 2) 
                                & (time_series.index.day == 29)].values
        time_series = time_series.drop(inds_to_drop)
      return time_series
      
        
    def get_seasonal_weights(self, ts, period):
      """
      Calculates seasonality using mean values of periodic fluctuations compared 
      to overall mean
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
      compared to overall mean of ts
      q: quantile to use
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
        calc_func=get_seasonal_weights):
      """
      Initialises a model with given output frequency
      """
      super().__init__(output_freq, stype='multiplicative')
      self.calculation_func = calc_func
      self.name = 'SeasonalModel ' + self.output_freq
      
    def train(self, time_series, verbose=False):
      """
      This function calculates seasonality indices for all possible input 
      frequences suitable for output_freq, and saves them to self.seasonal_weghts
      """
      if time_series.index.freq != self.output_freq:
          raise Exception('training dataset must have same \
          frequency as output series')
      base_ts = time_series
      self.seasonal_weghts = dict()
      # For each useful input frequency for the output frequency
      for freq_name, freq in mappings[self.output_freq]:
        curr_ts = base_ts.copy()
        # We get aligned data, drop leap day indices and then calculate seasonality
        curr_ts = self.check_dates_align(curr_ts, freq_name)
        
        # If number of periods for one cycle of seasonality is greater than
        # number of periods in input time series
        if freq > len(time_series):
          self.seasonal_weghts.update({
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
        self.seasonal_weghts.update({
            freq_name : self.calculation_func(self, curr_ts, freq)
        }) 
        
    def predict(self, time_series):
      """
      Converts lower-frequency time_series into higher-frequency, applying a model of
      frequency magnification separately to each term of the series and combining 
      results into a single sequence 
      """ 
      prediction = UniformModel.predict(self, time_series)
      prediction.rename('prediction', inplace=True)
      input_freq = self.input_freq
      if (self.output_freq in ['m', 'q']) and (input_freq in ['m', 'q']):
        input_freq = 'y'
      if input_freq in [item[0] for item in mappings[self.output_freq]]:
        aligned_index = self.check_dates_align(prediction, input_freq).\
            index
        l = len(self.seasonal_weghts[input_freq])
        inds_np = np.tile(
            self.seasonal_weghts[input_freq].index.values,
            int(np.ceil(len(aligned_index)/l))
        )

        new_inds = pd.Series(inds_np[:len(aligned_index)], index = aligned_index).rename('aligned')
        pred_w_inds = pd.concat([prediction, new_inds], axis = 1).dropna()
        seasonality = self.seasonal_weghts[input_freq].rename('seasonality')
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
