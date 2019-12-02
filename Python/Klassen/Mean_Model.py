import json
from warnings import warn

import numpy as np

from helperfunctions import rmse, mae


class MeanModel:
    def __init__(self, config_file=None):
        if config_file is not None:
            self.configs = json.load(open(config_file, 'r'))
        else:
            warn("No Config File given")

        self.last_means = None

    def fit(self, x, y, dic=None, **kwargs):
        if y is not None:
            self.last_mean = np.empty((y.shape[-1],))
        if dic is not None:
            self.last_means = {}
            for _id, frames in dic.items():
                self.last_means[_id] = {}
                for count, data in frames.items():
                    self.last_means[_id][count] = np.array(data['y'])
        else:
            print("No Data")

    def predict(self, steps=1, dic=None, **kwargs):
        if dic is not None:
            predictions = {}
            count = 0
            for _id, frames in dic.items():
                if _id not in self.last_means.keys():
                    self.last_means[_id] = {}
                for nr, data in frames.items():

                    y_ = np.array(data['y'])
                    if nr not in self.last_means[_id].keys():
                        self.last_means[_id][nr] = np.zeros((steps,) + y_.shape[1:])

                    mean = np.mean(self.last_means[_id][nr])
                    prediction = np.empty(y_.shape)
                    prediction[:] = np.NaN
                    for i in range(steps, len(y_) + 1, steps):
                        prediction[i - steps:i] = mean
                        self.last_means[_id][nr] = np.append(self.last_means[_id][nr],
                                                             y_[i - steps:i], axis=0)
                        mean = np.mean(self.last_means[_id][nr])
                    start = int(len(y_) / steps) * steps
                    end = len(y_) - 1
                    s = len(y_) % steps
                    if not s == 0:
                        prediction[start:end] = mean

                    predictions[count] = {'id': _id,
                                          'x': np.array(data['x']),
                                          'y': y_,
                                          'pred': prediction}
                    count += 1
            return predictions

    def evaluate(self, dic=None, scaler=None, steps=1, verbose=0, **kwargs):
        if dic is not None:
            errors = {}
            predictions = self.predict(steps=steps, dic=dic)
            for i, prediction in predictions.items():
                ypred = prediction['pred']
                ytrue = prediction['y'][:len(ypred)]
                n_features = ytrue.shape[-1]

                mask = ~np.isnan(ypred)
                ypred = ypred[mask].reshape((np.count_nonzero(mask), n_features))
                ytrue = ytrue[mask].reshape((np.count_nonzero(mask), n_features))
                errors[i] = {'RMSE_{}'.format(len(ytrue)): rmse(ypred, ytrue, scaler),
                             'RMSE_{}(x>180x<70)'.format(len(ytrue)): rmse(ypred, ytrue, scaler, (180, 70)),
                             'MAE_{}'.format(len(ytrue)): mae(ypred, ytrue, scaler),
                             'MAE_{}(180>x>70)'.format(len(ytrue)): mae(ypred, ytrue, scaler, (180, 70))}
            return predictions, errors
