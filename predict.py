from __future__ import absolute_import, division, print_function

import matplotlib
matplotlib.use('TkAgg')
import pathlib

import pandas
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers


class AutoSet:
    def __init__(self):
        self.url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
        self.columns = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
        self.dataset = pandas.read_csv(self.url, names=self.columns,  na_values="?", comment='\t', sep=" ", skipinitialspace=True)

    def drop_missing_rows(self):
        self.dataset = self.dataset.dropna()

# Tail the data
auto_set = AutoSet()
print(auto_set.dataset.tail())

# Clean the data, since column Horsepower contains 6 errors (?)
auto_set.drop_missing_rows()
