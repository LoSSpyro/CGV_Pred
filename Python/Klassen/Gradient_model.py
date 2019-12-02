import json
from warnings import warn
import numpy as np

from helperfunctions import rmse, mae


class GradientModel:
    def __init__(self, config_file=None):
        if config_file is not None:
            self.configs = json.load(open(config_file, 'r'))
        else:
            warn("No Config File given")

        self.last_grad = 0
        self.last_value = 0

    def fit(self, x=None, y=None, dic=None, **kwargs):
        if dic is not None:
            lv_means = list()
            grad_means = list()

            for _id, frames in dic.items():
                for count, data in frames.items():
                    y_ = np.array(data['y'])
                    if y_.shape[-1] > 1:
                        raise Exception("No support for multiple Features implemented")
                    lv_means.append(np.mean(y_, axis=0))
                    grad_means.append(np.mean(np.gradient(y_, axis=0)))

            self.last_value = np.mean(np.array(lv_means))
            self.last_grad = np.mean(np.array(grad_means))
            return

        self.last_value = np.empty((x.shape[2], 1))
        self.last_grad = np.empty((x.shape[2], 1))
        for i in range(x.shape[2]):
            self.last_value[i] = np.mean(x[:, :, i])
            self.last_grad[i] = np.mean(np.gradient(x[:, :, i]))

    def predict(self, steps=1, y=None, dic=None, **kwargs):
        if dic is not None:
            count = 0
            predictions = {}
            for _id, frames in dic.items():
                for _, data in frames.items():
                    lv = self.last_value
                    lg = self.last_grad
                    y_ = np.array(data['y'])
                    prediction = np.empty(y_.shape)
                    prediction[:] = np.NaN
                    for i in range(steps, len(y_) + 1, steps):
                        for j in range(i - steps, i, 1):
                            prediction[j:j + 1] = lv + lg
                            lv = lv + lg
                        lv = y_[i - 1]
                        lg = np.gradient(y_[i - steps:i], axis=0)[-1]
                    if not len(y_) % steps == 0:
                        start = int(len(y_) / steps) * steps
                        end = len(y_)
                        for i in range(len(y_) % steps):
                            prediction[i] = lv + lg
                            lv = lv + lg
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
                lg = self.last_grad
                for j in range(steps, len(y_i) + 1, steps):
                    for k in range(steps):
                        prediction[i, j - steps + k, :] = self.last_value + self.last_grad
                        self.last_value = self.last_value + self.last_grad
                    self.last_value = y[i, j - steps:j, :][-1]
                    self.last_grad = np.gradient(y[i, j - steps:j, -1])[-1:]

                if not len(y_i) % steps == 0:
                    start = int(len(y_i) / steps) * steps
                    end = len(y_i)
                    s = len(y_i) % steps
                    for j in range(start + s, len(y_i) + 1, s):
                        for k in range(s):
                            prediction[i, j - s + k, :] = self.last_value + self.last_grad
                            self.last_value = self.last_value + self.last_grad

                self.last_value = lv
                self.last_grad = lg

            return prediction

        prediction = np.empty((0,))

        for i in range(steps):
            prediction = np.append(prediction, self.last_value + self.last_grad)
            prediction = prediction.reshape(len(prediction), 1)

            if len(prediction) > 1:
                self.last_grad = np.gradient(prediction.reshape(len(prediction), ))[-1]

            self.last_value = prediction[-1]

        if y is not None:
            self.last_value = y[steps - 1]
            self.last_grad = np.gradient(y.reshape((len(y),)))[steps - 1]

        return prediction

    def evaluate(self, train=None, test=None, dic=None, steps=1, scaler=None, error_metric='rmse', verbose=0, **kwargs):
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
                             'MAE_{}(180>x>70'.format(len(ytrue)): mae(ypred, ytrue, scaler, (180, 70))}
            return predictions, errors


        last_value = train[-1]
        last_grad = np.gradient(train)[-1]

        test_grad = np.gradient(test)

        predictions = np.empty((0,))

        for i in range(steps, len(test) + 1, steps):
            for j in range(steps):
                predictions = np.append(predictions, last_value + last_grad)

                if len(predictions) > 1:
                    last_grad = np.gradient(predictions)[-1]
                last_value = predictions[-1]

            last_value = test[i - 1]
            last_grad = test_grad[i - 1]

        if error_metric == 'rmse':
            error_function = rmse
        else:
            raise NotImplementedError("No known Error Function {}".format(error_metric))

        error = error_function(test, predictions)

        return predictions, error
