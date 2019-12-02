from collections import defaultdict
from datetime import timedelta

import numpy as np
from matplotlib.dates import date2num
from pandas import DataFrame
from scipy.interpolate import splrep, splev

from helperclasses import ProgressBar
from helperfunctions import scale_dataframe, mae, rmse


class OpenAPSModel:
    def __init__(self, devicestatus):
        self.devicestatus = devicestatus

    def fit(self, x, y, **kwargs):
        pass

    def predict(self, x=None, steps=1, dic=None, in_features=None, out_features=None, scaler=None, verbose=0, **kwargs):

        if out_features is None:
            out_features = ['sgv']
        if in_features is None:
            in_features = ['sgv']

        def nan_array():
            return np.full(x.shape, np.NaN)

        if verbose == 1 and x is not None:
            pb = ProgressBar("Extract Predictions from Device Status", x.shape[0])

        if dic is not None:
            empty_pred = {}
            c = 0
            for _id, frames in dic.items():
                for _, data in frames.items():
                    df = data['df']
                    nan_pred = np.empty(df[out_features].values.shape)
                    nan_pred[:] = np.NaN
                    empty_pred[c] = {'id': _id,
                                     'y': df[out_features].values,
                                     'pred': nan_pred}
                    c += 1

            predictions = {}
            count, last_finding = 0, -1
            for _id, frames in dic.items():
                for index, data in frames.items():
                    df = data['df']
                    valid_status = [status for status in self.devicestatus[_id]
                                    if df.index[-1] > status['date'] > df.index[0]
                                    and 'predBGs' in status.keys()]

                    dates = set()

                    if len(valid_status) == 0:
                        count += 1
                        continue

                    print("Found OpenAPS Pred. at {} #{}".format(_id, index))

                    def default_df():
                        p_df = DataFrame(index=df.index)
                        for feature in in_features:
                            p_df[feature] = np.NaN
                        return p_df

                    pred_df = defaultdict(default_df)

                    for status in valid_status:
                        date = status['date']
                        if date in dates:
                            continue
                        closest_date = df.index[df.index.get_loc(date, method='nearest')]
                        for key, preds in status['predBGs'].items():
                            if key.endswith("true"):
                                continue

                            x_ax = np.array([date2num(date + timedelta(minutes=5 * (i + 1)))
                                             for i in range(len(preds))])
                            y = np.array(preds)
                            spl = splrep(x_ax, y)

                            try:
                                pred = splev(date2num(closest_date + timedelta(minutes=5 * steps)), spl)
                            except:
                                continue
                            pred_df[key].loc[closest_date] = pred
                            dates.add(date)
                    pred_df_keys = pred_df.keys()
                    keys = set(predictions.keys())
                    keys.update(pred_df_keys)

                    for key in keys:
                        dataframe = DataFrame(pred_df[key])
                        if scaler is not None:
                            dataframe = scale_dataframe(dataframe, scaler)

                        try:
                            predictions[key][count] = {'id': _id,
                                                       'y': df[out_features].values,
                                                       'pred': dataframe[out_features].values}
                            if not last_finding == count-1:
                                for i in range(last_finding, count):
                                    predictions[key][i] = {'id': empty_pred[i]['id'],
                                                           'y': empty_pred[i]['y'],
                                                           'pred': empty_pred[i]['pred']}
                        except KeyError:
                            if count > 0:
                                for i in range(count):
                                    tmp_pred = np.empty(empty_pred[i]['y'].shape)
                                    tmp_pred[:] = np.NaN
                                    try:
                                        predictions[key][i] = {'id': empty_pred[i]['id'],
                                                               'y': empty_pred[i]['y'],
                                                               'pred': tmp_pred}
                                    except KeyError:
                                        predictions[key] = {i: {'id': empty_pred[i]['id'],
                                                                'y': empty_pred[i]['y'],
                                                                'pred': tmp_pred}}

                                predictions[key][count] = {'id': _id,
                                                           'y': df[out_features].values,
                                                           'pred': dataframe[out_features].values}
                            else:
                                predictions[key] = {count: {'id': _id,
                                                            'y': df[out_features].values,
                                                            'pred': dataframe[out_features].values}}

                    #print({"Count: {}. In {}; #{}".format(count, _id, )}"Count:", count, "Last Finding:", last_finding, end="\t\t\t\t\r")
                    last_finding = count
                    count += 1
            if not last_finding == count-1:
                for i in range(last_finding, count):
                    predictions[key][i] = {'id': empty_pred[i]['id'],
                                           'y': empty_pred[i]['y'],
                                           'pred': empty_pred[i]['pred']}
            return predictions

    def evaluate(self, x, y, dic=None,
                 in_features=None, out_features=None,
                 scaler=None, verbose=0, **kwargs):

        if in_features is None:
            in_features = ['sgv']
        if out_features is None:
            out_features = ['sgv']

        # print(kwargs)

        predictions = kwargs['predictions'] if 'predictions' in kwargs else self.predict(x=x, y=y, dic=dic,
                                                                                         in_features=in_features,
                                                                                         out_features=out_features,
                                                                                         scaler=scaler,
                                                                                         verbose=verbose, **kwargs)
        errors = {}
        for label, value in predictions.items():
            errors[label] = {}
            for i, prediction in value.items():
                ypred = prediction['pred']
                ytrue = prediction['y'][:len(ypred)]
                mask = ~np.isnan(ypred)
                try:
                    ypred = ypred[mask].reshape((np.count_nonzero(mask), len(out_features)))
                    ytrue = ytrue[mask].reshape((np.count_nonzero(mask), len(out_features)))
                except Exception as e:
                    print("Shapes:", ypred.shape, ytrue.shape, mask.shape)
                    print("Lengths:", len(ypred), len(ytrue), len(mask))
                    print("Label:", label, "i:", i)
                    raise e
                errors[label][i] = {'RMSE_{}'.format(len(ytrue)): rmse(ypred, ytrue, scaler),
                                    'RMSE_{}(x>180x<70)'.format(len(ytrue)): rmse(ypred, ytrue, scaler, (180, 70)),
                                    'MAE_{}'.format(len(ytrue)): mae(ypred, ytrue, scaler),
                                    'MAE_{}(180>x>70)'.format(len(ytrue)): mae(ypred, ytrue, scaler, (180, 70))}
        return predictions, errors
