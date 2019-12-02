from collections import defaultdict

from pyramid.arima import auto_arima
import numpy as np

from helperclasses import ProgressBar
from helperfunctions import rmse, rmse_weighted, mae
import json
import multiprocessing as mp


class ArimaModel:
    def __init__(self, config_file=None, arima_model=None):
        if arima_model is not None:
            self.models = arima_model.models
            self.configs = arima_model.configs
        else:
            self.models = None
            self.configs = json.load(open(config_file, 'r'))

    def fit(self, x, y, dic=None, verbose=0, **kwargs):

        scope = self.configs['fitting']

        seasonal = scope['seasonal'] if 'seasonal' in scope else False
        split = scope['split'] if 'split' in scope else 0.66

        min_p = scope['min_p'] if 'min_p' in scope else 0
        max_p = scope['max_p'] if 'max_p' in scope else 5
        min_q = scope['min_q'] if 'min_q' in scope else 0
        max_q = scope['max_q'] if 'max_q' in scope else 5
        d = scope['d'] if 'd' in scope else None
        max_d = scope['max_d'] if 'max_d' in scope else 2

        min_P = scope['min_P'] if 'min_P' in scope else 1
        max_P = scope['max_P'] if 'max_P' in scope else 2
        min_Q = scope['min_Q'] if 'min_Q' in scope else 1
        max_Q = scope['max_Q'] if 'max_Q' in scope else 2
        D = scope['D'] if 'D' in scope else None
        max_D = scope['max_D'] if 'max_D' in scope else 1

        error_action = scope['error_action'] if 'error_action' in scope else 'ignore'
        suppress_warnings = scope['suppress_warnings'] if 'suppress_warnings' in scope else True
        stepwise = scope['stepwise'] if 'stepwise' in scope else True

        if dic is not None:
            self.models = {}
            for _id, frames in dic.items():
                self.models[_id] = {}
                for i, data in frames.items():
                    y_ = np.array(data['y'])
                    self.models[_id][i] = auto_arima(y_,
                                                     start_p=min_p, d=d, start_q=min_q,
                                                     max_p=max_p, max_d=max_d, max_q=max_q,
                                                     start_P=min_P, D=D, start_Q=min_Q,
                                                     max_P=max_P, max_D=max_D, max_Q=max_Q,
                                                     seasonal=seasonal,
                                                     stepwise=stepwise,
                                                     error_action=error_action,
                                                     suppress_warnings=suppress_warnings)
        else:
            print("No Data")

    def __cross_validate__(self, x, model, steps):
        sp = int(len(x) * 0.66)
        errors = np.empty((0,))
        for sample in x:
            train, test = sample[:sp, 0], sample[sp:, 0]

            history = train
            yhat = np.empty((0,))
            for i in range(0, len(test), steps):
                yhat = np.append(yhat, model.fit_predict(history, n_periods=steps))
                history = np.append(history, test[i:i + steps])

            if np.isnan(yhat).any():
                raise Exception("Nan Predictions")

            errors = np.append(errors, rmse(test, yhat))

        return model, np.mean(errors)

    def predict(self, steps, y=None, dic=None, model=None, history=None, verbose=0, **kwargs):
        scope = self.configs['fitting']

        seasonal = scope['seasonal'] if 'seasonal' in scope else False
        split = scope['split'] if 'split' in scope else 0.66

        min_p = scope['min_p'] if 'min_p' in scope else 0
        max_p = scope['max_p'] if 'max_p' in scope else 5
        min_q = scope['min_q'] if 'min_q' in scope else 0
        max_q = scope['max_q'] if 'max_q' in scope else 5
        d = scope['d'] if 'd' in scope else None
        max_d = scope['max_d'] if 'max_d' in scope else 2

        min_P = scope['min_P'] if 'min_P' in scope else 1
        max_P = scope['max_P'] if 'max_P' in scope else 2
        min_Q = scope['min_Q'] if 'min_Q' in scope else 1
        max_Q = scope['max_Q'] if 'max_Q' in scope else 2
        D = scope['D'] if 'D' in scope else None
        max_D = scope['max_D'] if 'max_D' in scope else 1

        error_action = scope['error_action'] if 'error_action' in scope else 'ignore'
        suppress_warnings = scope['suppress_warnings'] if 'suppress_warnings' in scope else True
        stepwise = scope['stepwise'] if 'stepwise' in scope else True

        if dic is not None:
            predictions = {}
            count = 0
            for _id, frames in dic.items():
                if _id in self.models.keys():
                    for i, data in frames.items():
                        y_ = data['y']
                        preds = np.empty(y_.shape)
                        preds[:] = np.NaN

                        model = self.models[_id][i]
                        for j in range(steps, len(y_) + 1, steps):
                            preds[j - steps:j] = model.predict(steps).reshape(steps, 1)
                            model.add_new_observations(y_[j - steps:j, 0])
                        try:
                            remainder = len(y_) % steps
                            rest_length = int(len(y_) / steps) * steps
                            if remainder > 0:
                                p = model.predict(remainder).reshape(remainder, 1)
                                preds[rest_length:len(y_)] = p
                        except Exception as e:
                            print("Exception", e, "\n",
                                  "remainder:", remainder, "rest length:", rest_length, "\n",
                                  "p.shape", p.shape, "preds.shape", preds.shape, "\n",
                                  "rest length shape", preds[rest_length:len(y_)].shape)
                        if np.any(np.isnan(preds)):
                            raise ValueError("Nan Prediction ID:{} Nr.{}".format(_id, count))

                        predictions[count] = {'id': _id,
                                              'y': y_,
                                              'pred': preds}
                        count += 1

                else:
                    for _, data in frames.items():
                        y_ = np.array(data['y'])
                        sp = int(len(y_) * split)
                        train, test = y_[:sp], y_[sp:]
                        preds = np.empty(test.shape)
                        preds[:] = np.NaN
                        history = train[:, 0]
                        model = auto_arima(history,
                                           start_p=min_p, d=d, start_q=min_q,
                                           max_p=max_p, max_d=max_d, max_q=max_q,
                                           start_P=min_P, D=D, start_Q=min_Q,
                                           max_P=max_P, max_D=max_D, max_Q=max_Q,
                                           seasonal=seasonal,
                                           stepwise=stepwise,
                                           error_action=error_action,
                                           suppress_warnings=suppress_warnings)

                        for i in range(steps, len(test) + 1, steps):
                            preds[i - steps:i] = model.predict(steps).reshape(steps, 1)
                            model.add_new_observations(test[i - steps:i, 0])

                        try:
                            remainder = len(test) % steps
                            rest_length = int(len(test) / steps) * steps
                            if remainder > 0:
                                p = model.predict(remainder).reshape(remainder, 1)
                                preds[rest_length:len(test)] = p
                        except Exception as e:
                            print("Exception", e, "\n",
                                  "remainder:", remainder, "rest length:", rest_length, "\n",
                                  "p.shape", p.shape, "preds.shape", preds.shape, "\n",
                                  "rest length shape", preds[rest_length:len(test)].shape)
                        if np.any(np.isnan(preds)):
                            raise ValueError("Nan Prediction ID:{} Nr.{}".format(_id, count))

                        predictions[count] = {'id': _id,
                                              'y': y_,
                                              'pred': preds}
                        count += 1
            return predictions

        elif y is not None:
            if len(y.shape) == 3:
                sp = int(y.shape[1] * split)
                train, test = y[:, :sp, :], y[:, sp:, :]
                predictions = np.empty(test.shape)

                if verbose == 1:
                    pb = ProgressBar("Predicting ARIMA", len(y))

                for i, x in enumerate(train):

                    history = x[:, 0]
                    model = auto_arima(history,
                                       start_p=min_p, d=d, start_q=min_q,
                                       max_p=max_p, max_d=max_d, max_q=max_q,
                                       start_P=min_P, D=D, start_Q=min_Q,
                                       max_P=max_P, max_D=max_D, max_Q=max_Q,
                                       seasonal=seasonal,
                                       stepwise=stepwise,
                                       error_action=error_action,
                                       suppress_warnings=suppress_warnings)

                    if verbose == 1:
                        pb.update(i + 1)

                    for j in range(steps, len(test[i]) + 1, steps):
                        predictions[i, j - steps:j, :] = model.predict(steps).reshape(steps, 1)
                        model.add_new_observations(test[i, j - steps:j, 0])

                    remainder = len(test[i]) % steps
                    rest_length = int(len(test[i]) / steps) * steps
                    predictions[i, rest_length:len(test[i]), :] = model.predict(remainder).reshape(remainder, 1)
                    if np.any(np.isnan(predictions)):
                        raise ValueError("Nan Prediction {}".format(i))

                if verbose == 1:
                    pb.done()

                return predictions

        if model is None:
            model = self.model

        if history is not None:
            model.fit(history)

        return model.predict(steps)

    def evaluate(self, steps=1, dic=None, scaler=None, error_metric='rmse', verbose=0, **kwargs):
        if error_metric == 'rmse':
            error_function = rmse
        elif error_metric == 'rmse_weighted':
            error_function = rmse_weighted
        else:
            raise NotImplementedError("Did not found error_metric: {}".format(error_metric))

        if dic is not None:
            predictions = self.predict(steps=steps, dic=dic, verbose=verbose, **kwargs)
            error = {}
            for i, prediction in predictions.items():
                ypred = prediction['pred']
                ytrue = prediction['y'][:len(ypred)]
                mask = ~np.isnan(ypred)
                ypred = ypred[mask].reshape((np.count_nonzero(mask),) + (1,))
                ytrue = ytrue[mask].reshape((np.count_nonzero(mask),) + (1,))
                error[i] = {'RMSE_{}'.format(len(ytrue)): rmse(ypred, ytrue, scaler),
                            'RMSE_{}(x>180x<70)'.format(len(ytrue)): rmse(ypred, ytrue, scaler, (180, 70)),
                            'MAE_{}'.format(len(ytrue)): mae(ypred, ytrue, scaler),
                            'MAE_{}(180>x>70'.format(len(ytrue)): mae(ypred, ytrue, scaler, (180, 70))}

            return predictions, error
        else:
            raise NotImplementedError("Not implemented if dic == None")
