import json
from warnings import warn

import numpy as np

from helperfunctions import rmse, mae


class LastValueModel:
    def __init__(self, config_file=None):
        if config_file is not None:
            self.configs = json.load(open(config_file, 'r'))
        else:
            warn("No Config File given")

        self.last_value = None

    def fit(self, x, y, dic=None, **kwargs):
        if y is not None:
            self.last_value = np.empty((y.shape[-1],))
        if dic is not None:
            means = list()
            for _id, frames in dic.items():
                for count, data in frames.items():
                    means.append(np.mean(np.array(data['y']), axis=0))
            self.last_value = np.mean(np.array(means), axis=0)

    def predict(self, steps=1, y=None, dic=None, **kwargs):
        if dic is not None:
            predictions = {}
            count = 0
            for _id, frames in dic.items():
                for _, data in frames.items():
                    lv = self.last_value

                    y_ = np.array(data['y'])
                    prediction = np.empty(y_.shape)
                    prediction[:] = np.NaN
                    for i in range(steps, len(y_) + 1, steps):
                        prediction[i - steps:i] = lv
                        lv = y_[i - 1]
                    start = int(len(y_) / steps) * steps
                    end = len(y_) - 1
                    s = len(y_) % steps
                    if not s == 0:
                        prediction[start:end] = lv

                    predictions[count] = {'id': _id,
                                          'x': np.array(data['x']),
                                          'y': y_,
                                          'pred': prediction}
                    count += 1
            return predictions

        if len(y.shape) == 3:
            prediction = np.empty(y.shape)
            for i, y_i in enumerate(y):
                lv = self.last_value
                for j in range(steps, len(y_i) + 1, steps):
                    prediction[i, j - steps:j, :] = self.last_value

                    self.last_value = y[i, j - steps:j, :][-1]

                if not len(y_i) % steps == 0:
                    start = int(len(y_i) / steps) * steps
                    end = len(y_i)
                    s = len(y_i) % steps
                    if i == 2165:
                        print("End of TS lastValue:", self.last_value, y[i, j - steps:j, :][-1])
                    for j in range(start + s, len(y_i) + 1, s):
                        if i == 2165:
                            print("i, j-s, j:", i, j - s, j)
                        prediction[i, j - s:j, :] = self.last_value
                self.last_value = lv

            return prediction

        if self.last_value is None:
            self.last_value = 0

        if y is not None:
            shape = (steps, y.shape[-1])
        else:
            shape = (steps, 1)
        prediction = np.zeros(shape)
        prediction[:] = self.last_value

        if y is not None:
            self.last_value = y[steps - 1]

        return prediction

    def evaluate(self, train=None, test=None, dic=None, scaler=None, steps=1, error_metric='rmse', verbose=0, **kwargs):
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

        last_value = train[-1]

        predictions = np.empty((0,))
        for i in range(steps, len(test) + 1, steps):
            if verbose == 1:
                if i % 20 == 0:
                    print("it: {}".format(i))
            pred = np.zeros((steps,))
            pred[:] = last_value
            predictions = np.append(predictions, pred)
            last_value = test[i - 1]

        if error_metric == 'rmse':
            error_function = rmse
        else:
            raise NotImplementedError("No known Error Function {}".format(error_metric))

        error = error_function(test, predictions)

        return error
