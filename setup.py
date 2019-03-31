import pandas as pd
from sklearn.model_selection import train_test_split


class Setup:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._path = kwargs['path']
        self._features = kwargs['features']
        self._axis = self.get_arg('axis', 0)
        self._random_state = self.get_arg('random', 0)

        self._load_data()

    def get_arg(self, arg, default):
        return arg if arg in self.kwargs else default

    def _load_data(self):
        self.raw_data = pd.read_csv(self._path)
        self.data = self.raw_data.dropna(axis=self._axis)

    def split_data(self):
        y = self.data.Price
        X = self.data[self._features]
        return train_test_split(X, y, random_state=self._random_state)
