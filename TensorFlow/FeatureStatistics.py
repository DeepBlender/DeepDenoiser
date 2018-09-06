class FeatureStatistics:
  def __init__(self, number_of_channels, statistics, statistics_log1p):
    self.number_of_channels = number_of_channels
    self.statistics = statistics
    self.statistics_log1p = statistics_log1p

class Statistics:
  def __init__(self, minimum, maximum, mean, variance, coverage):
    self.minimum = minimum
    self.maximum = maximum
    self.mean = mean
    self.variance = variance
    self.coverage = coverage
