import pickle
from copy import deepcopy

import matplotlib.dates as md
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import keras.backend as K

import random

from collections import defaultdict
from datetime import timedelta, datetime

from bokeh.plotting import figure
from bokeh.models import Panel
from pandas import DataFrame, concat, Index
from sklearn.preprocessing import MinMaxScaler

from helperclasses import NearestNeighbourDateList
from matplotlib.dates import date2num
from scipy.interpolate import interp1d
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller


def roundTime(dt=None, roundTo=60):
    """Round a datetime object to any time lapse in seconds
    dt : datetime.datetime object, default now.
    roundTo : Closest number of seconds to round to, default 1 minute.
    Author: Thierry Husson 2012 - Use it as you want but don't blame me.
    """
    if dt is None:
        dt = datetime.now()
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds + roundTo / 2) // roundTo * roundTo
    return dt + timedelta(0, rounding - seconds, -dt.microsecond)


def interpolateEntries(start_entry, current_entry, new_time_step, end_entry, kind="linear"):
    if kind == "linear":
        total_diff = abs((start_entry['date'] - end_entry['date']).total_seconds())
        start_diff = abs((start_entry['date'] - new_time_step).total_seconds())
        end_diff = abs((new_time_step - end_entry['date']).total_seconds())

        return start_diff / total_diff * end_entry['sgv'] + end_diff / total_diff * start_entry['sgv']
    elif kind == 'quadratic':
        x = [date2num(start_entry['date']), date2num(current_entry['date']), date2num(end_entry['date'])]
        y = [start_entry['sgv'], current_entry['sgv'], end_entry['sgv']]
        f = interp1d(x, y, kind=kind)
        return f(date2num(new_time_step))

    else:
        return None


def getPlotData(entries):
    return [entry['date'] for entry in entries], [entry['sgv'] for entry in entries]


def plotData(data):
    try:
        dates, values = data
    except:
        dates, values = getPlotData(data)
    ax = plt.gca()
    xfmt = md.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(xfmt)
    plot_label = str(dates[-1]) + " - " + str(dates[0])
    plt.plot(dates, values)
    plt.gcf().autofmt_xdate()
    plt.ylim(0, max(values) + 20)
    plt.axhspan(70, 200, alpha=0.1, color='green')
    plt.title(plot_label)
    plt.show()


def plotDataFrame(df, method=None, order=None, figsize=(12, 6), dpi=100):
    base = getBase(df.index.tolist()[1])
    if not method:
        data = df.resample('300S', base=base).mean()
    else:
        data = df.resample('300S', base=base).interpolate(method=method, order=order)
    plt.figure(figsize=figsize, dpi=dpi)
    data.plot()
    plt.show()


def getDependantDates(source):
    dependantTimeSteps = defaultdict(NearestNeighbourDateList)

    for entry in source:
        date = entry['date']
        rounded_time = roundTime(date, roundTo=60 * 5)
        if rounded_time < date:
            prev_time_step = rounded_time
            next_time_step = rounded_time + timedelta(minutes=5)
        else:
            next_time_step = rounded_time
            prev_time_step = rounded_time - timedelta(minutes=5)

        dependantTimeSteps[next_time_step].append(entry, next_time_step)
        dependantTimeSteps[prev_time_step].append(entry, prev_time_step)

    return dependantTimeSteps


def interpolateTimeSteps(source):
    values = {}
    for ts in source.keys():
        dependant_time_steps_list = source[ts]
        if len(dependant_time_steps_list) > 2:
            raise Exception("Too many dependant timesteps!!!")
        if len(dependant_time_steps_list) < 2:
            warnings.warn("There need to be more Timesteps for Interpolation")
        else:
            if (ts - dependant_time_steps_list[0]['date']).total_seconds() < 0:
                prev_entry = dependant_time_steps_list[1]
                next_entry = dependant_time_steps_list[0]
            else:
                prev_entry = dependant_time_steps_list[0]
                next_entry = dependant_time_steps_list[1]

                # Interpolate between the timestep before and the timestep after the given Moment
            prev_diff = float(abs((ts - prev_entry['date']).total_seconds()))
            next_diff = float(abs((ts - next_entry['date']).total_seconds()))
            total_diff = prev_diff + next_diff

            values[ts] = prev_diff / total_diff * next_entry['sgv'] + next_diff / total_diff * prev_entry['sgv']

    # try to get values for dependetTimesteps with only 1 dependant
    for ts in source.keys():
        dependant_time_steps_list = source[ts]
        if len(dependant_time_steps_list) == 1:
            first_entry = dependant_time_steps_list[0]

            if (ts - first_entry['date']).total_seconds() > 0:
                other_time_step = ts + timedelta(minutes=5)
                if other_time_step in values:
                    prev_diff = float(abs(ts - first_entry['date']).total_seconds())
                    next_diff = float(abs(ts - other_time_step).total_seconds())
                    total_diff = prev_diff + next_diff

                    values[ts] = prev_diff / total_diff * values[other_time_step] + next_diff / total_diff * \
                                 first_entry['sgv']
                else:
                    warnings.warn("There are still values with only 1 dependant timestep!")
            else:
                other_time_step = ts - timedelta(minutes=5)
                if other_time_step in values:
                    next_diff = float(abs(ts - first_entry['date']).total_seconds())
                    prev_diff = float(abs(ts - other_time_step).total_seconds())
                    total_diff = prev_diff + next_diff

                    values[ts] = prev_diff / total_diff * first_entry['sgv'] + next_diff / total_diff * values[
                        other_time_step]
                else:
                    warnings.warn("There are still values with only 1 dependant timestep!")

    return values


def roundedSeconds(date):
    if date.second < 30:
        return -timedelta(seconds=date.second, microseconds=date.microsecond)
    else:
        return timedelta(seconds=60.0 - date.second, microseconds=-date.microsecond)


def getBase(date):
    return (date.minute % 5) * 60 + date.second


def forecast_accuracy(forecast, actual):
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs(forecast - actual) / np.abs(actual))
    # Mean Error
    me = np.mean(forecast - actual)
    # Mean Absolute Error
    mae = np.mean(np.abs(forecast - actual))
    # Mean Percentage Error
    mpe = np.mean((forecast - actual) / actual)
    # Root Mean Squared Error
    rmse = np.mean((forecast - actual) ** 2) ** .5
    # Lag 1 Autocorrelation of Error
    acf1 = acf(forecast - actual)[1]
    # Correlation between the Actual and the Forecast
    corr = np.corrcoef(forecast, actual)[0, 1]
    # Min-Max Error
    mins = np.amin(np.hstack([forecast[:, None],
                              actual[:, None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:, None],
                              actual[:, None]]), axis=1)
    minmax = 1 - np.mean(mins / maxs)

    result = {'mape': mape, 'me': me, 'mae': mae, 'mpe': mpe, 'rmse': rmse, 'acf1': acf1, 'corr': corr,
              'minmax': minmax}
    # print("-"*20)
    # print(forecast, actual, sep="\n\n")
    # print("-"*20)

    return result


def choose_random(n, source):
    return random.sample(source, n)


def split_dataframe(df, n_obs, n_label):
    values = df.values

    x, y = list(), list()
    for i in range(len(values)):
        obs_end = i + n_obs
        label_end = obs_end + n_label
        if obs_end > len(values) - 1 or label_end > len(values) - 1:
            break
        seq_x, seq_y = np.asarray(values[i:obs_end]), values[label_end]
        x.append(seq_x)
        y.append(seq_y)
    return np.asarray(x), np.asarray(y)


def get_difference_order(data, conf=0.05):
    tmp_values = data.iloc[:, 0].values
    tmp_data = data
    result = adfuller(tmp_values)

    d = 0
    while result[1] > conf and d < 2:
        d += 1
        tmp_data = tmp_data.diff().dropna()
        tmp_values = tmp_data.iloc[:, 0].values
        result = adfuller(tmp_values)
    return d


def get_arimax_configs(min_p, min_q, max_p, max_q, d):
    configs = list()
    p_params = range(min_p, max_p + 1)
    q_params = range(min_q, max_q + 1)
    for p in p_params:
        for q in q_params:
            cfg = [(p, d, q)]
            configs.extend(cfg)

    print("# of Configs:", len(configs))
    for cfg in configs:
        print(cfg, end=" ")
    return configs


def get_sarimax_configs(seasonal, min_p, min_q, max_p, max_q, d):
    configs = list()
    p_params = range(min_p, max_p + 1)
    P_params = range(min_p, max_p + 1)
    q_params = range(min_q, max_q + 1)
    Q_params = range(min_q, max_q + 1)
    d_params = d
    D_params = [0, 1, 2]
    m_params = seasonal
    t_params = ['n', 'c', 't', 'ct']
    for p in p_params:
        for q in q_params:
            for t in t_params:
                for P in P_params:
                    for D in D_params:
                        for Q in Q_params:
                            for m in m_params:
                                cfg = [(p, d, q), (P, D, Q, m), t]
                                configs.append(cfg)

    print("# of Configs:", len(configs))
    for cfg in configs:
        print(cfg, end=" ")
    return configs


def plot_pred_vs_actual(train, test, pred, show_conf=True, figsize=(12, 5), dpi=100, loc='upper left', fontsize=8,
                        stepindices=None):
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(train, label="training")
    plt.plot(test, label="actual")
    plt.plot(pred, label="predictions")

    if not stepindices is None:
        for index in stepindices:
            plt.axvline(x=index, alpha=.15, linestyle='-')

    plt.axhspan(70, 200, alpha=0.1, color='green')
    plt.title('Forecast vs Actuals')
    plt.legend(loc=loc, fontsize=fontsize)
    plt.show()


def series_to_supervised(data, n_in, n_out, dropnan=True):
    """
    This function takes as input a dataframe of timesteps and splits the dataset in to pairs of observation-groundtruth.
    Their length depends on n_in and n_out
    :param data: The dataframe to extract Observations from
    :param n_in: The number of observations as prediction input
    :param n_out: The number of timesteps to forecast
    :param dropnan: Weather or not to drop NaN values from the resulting dataframes (Default: True)
    :return: Two Arrays; 1. Array: Observations, 2. Array ground-truth predictions of the observations
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d' % (j + 1, i)) for j in range(n_vars)]

    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)

    obs, labels = list(), list()
    for v in agg.values:
        ob, lab = list(), list()
        for i in range(n_vars):
            ob.append(v[i:n_in * n_vars][0::n_vars])
            labels.append(v[n_in * n_vars + i:][0::n_vars])

        obs.append(ob)

    return np.asarray(obs), np.asarray(labels)


def prepare_data(df, split, n_obs, n_labels, feature_range=(0, 1)):
    """
    This function prepares the data in regard of the split into train-, and testset. To make reasonable predictions
    the values have to get scaled to a specific data range, so that the NN can process the data
    :param df: The Dataframe which contains all observations
    :param split: the split in percent at which the Dataset is split between train- and testset
    :param n_obs: number of observations for one example
    :param n_labels: number of ground truth timesteps for one example
    :param feature_range: the range to which the values should get scaled to
    :return: scaler, (train_x, train_y),, (test_x, test_y)
    """

    if df.shape[1] > 1:
        raise NotImplementedError("Multivariate Forecast not yet supported!")

    scaler = MinMaxScaler(feature_range=feature_range)

    raw_values = df.values
    raw_df = df
    d_order = get_difference_order(df)

    for _ in range(d_order):
        raw_df = raw_df.diff().dropna()
        raw_values = raw_df.values

    scaled_values = scaler.fit_transform(raw_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)

    split_point = int(len(df) * split) - d_order

    supervised_x, supervised_y = series_to_supervised(scaled_values, n_obs, n_labels)

    train_x, train_y = supervised_x[:split_point], supervised_y[:split_point]
    test_x, test_y = supervised_x[split_point:], supervised_y[split_point:]

    return scaler, (train_x, train_y), (test_x, test_y)


def inverse_difference(last_ob, arr):
    """
    :param last_ob: last observations, from which the difference starts
    :param arr: an Array containing all values following the last observation
    :return: a list which has the original values restored
    """
    inverted = list()
    inverted.append(arr[0] + last_ob)

    for i in range(1, len(arr)):
        inverted.append((arr[i] + inverted[i - 1]))

    return inverted


def rescale(scaler, forecast):
    fc = np.array(forecast)
    fc = fc.reshape(1, len(fc))
    fc = scaler.inverse_transform(fc)
    return fc[0, :]


def revert_forecast(df, d, split, scaler, forecasts, n_obs):
    """
    This function kind of needs a specific setup to work: The forecasts can't be out of order and
    the examples need to be setup so that each following example is shifted by 1
    :param df:
    :param d:
    :param split:
    :param scaler:
    :param forecasts:
    :param n_obs:
    :return:
    """
    split_point = int(len(df) * split) - d

    inverted_fcs = list()
    for fc in forecasts:
        inverted_fcs.append(rescale(scaler, fc))

    sources = list()
    curr = df
    for _ in range(d):
        sources.insert(0, deepcopy(curr))
        curr = curr.diff().dropna()

    for i, source in enumerate(sources):
        new_forecasts = list()

        for j, fc in enumerate(inverted_fcs):
            last_ob = source.values[:, 0][split_point + n_obs + i + j]
            inverted = inverse_difference(last_ob, fc)
            new_forecasts.append(inverted)

        inverted_fcs = new_forecasts

    return inverted_fcs


def evaluate_forecasts(actuals, forecasts, n_labels, error_metric='rmse'):
    errors = list()
    for i in range(n_labels):
        actual = actuals[:, i]
        predicted = [fc[i] for fc in forecasts]
        error = forecast_accuracy(predicted, actual)[error_metric]
        errors.append(error)

    return errors


# make a persistence forecast
def persistence(last_ob, n_seq):
    return [last_ob for _ in range(n_seq)]


def hcf(numbers):
    smallest = numbers[0]
    for n in numbers:
        if n < smallest:
            smallest = n

    for i in range(smallest, 0, -1):
        if all(j % i == 0 for j in numbers):
            return i

    return None


def load_pkl_file(filename):
    obj = pickle.load(open(filename, 'rb'))
    return obj


def scale_dataframe(data, scalers):
    df = data.copy()
    for column, scaler in scalers.items():
        df[column] = data[column].values
        df[column] = df[[column]]
        if np.count_nonzero(~np.isnan(data[column].values)) == 0:
            continue
        null_index = df[column].isnull()
        df.loc[~null_index, [column]] = scaler.transform(df.loc[~null_index, [column]])
    return df


def convert_to_supervised(data, n_in=1, n_out=1, skips=0, dropnan=True):
    columns, names = list(), list()
    df = DataFrame(data)

    for i in range(n_in, 0, -1):
        columns.append(df.shift(i + skips))
        names += ['t-{}'.format(i + skips)]

    for i in range(0, n_out):
        columns.append(df.shift(-i))
        name = 't' if i == 0 else 't+{}'.format(i)
        names += [name]

    df = concat(columns, axis=1)
    df.columns = names
    if dropnan:
        df.dropna(inplace=True)
    return df


def get_divisors(number):
    divs = list()
    for i in range(number):
        if i == 0:
            continue
        if number % i == 0:
            divs.append(i)
    return divs


def pad_dataframe(df, pad_length, mask_value=-1):
    if pad_length <= 0:
        warnings.warn("Invalid pad_length of {}. No padding done".format(pad_length))
        return DataFrame(df)
    last_index = df.index[-1]
    index = list()
    for i in range(pad_length):
        index.append(last_index + timedelta(minutes=(i + 1) * 5))
    index = Index(index)

    tmpdf = DataFrame(df)

    padding = [mask_value for _ in range(pad_length)]
    padded_df = DataFrame()
    padded_df['data'] = padding
    padded_df.set_index(index, inplace=True)

    return concat([tmpdf, padded_df])


def roll_dataframe(df, window_size, skips):
    dataframes = list()
    for i in range(0, len(df), skips):
        loc = df.iloc[i:i + window_size]
        if len(loc) == window_size:
            dataframes.append(loc)
    return dataframes


def split_dataframe(df, threshold, minlen, remove=True, split_feature='sgv'):
    lower, upper = threshold
    if lower > upper:
        lower, upper = upper, lower

    tmp_df = DataFrame(index=df.index)
    tmp_df[split_feature] = df[split_feature].values
    ids = pd.isnull(tmp_df[tmp_df < upper][tmp_df > lower]).any(1).nonzero()[0]

    if len(ids) == 0:
        return [df]

    prevs = [0]
    prevs.extend(ids)

    dataframes = list()
    for prev, curr in zip(prevs, ids):
        if curr - prev <= minlen:
            continue

        dataframes.append(df.iloc[prev + 1:curr])

    if len(df) - ids[-1] > minlen:
        dataframes.append(df.iloc[ids[-1] + 1:len(df)])

    return dataframes


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def root_mean_squared_error_weighted(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) + K.sqrt(K.mean(K.abs((y_pred - y_true)) * K.square(y_true)))


def rmse_weighted(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true))) + np.sqrt(
        np.mean(np.abs((y_pred - y_true)) * np.square(y_true)))


def rmse(y_true, y_pred, scalers=None, threshold=None):
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.inf

    ypred = y_pred.copy()
    ytrue = y_true.copy()

    if scalers is not None:
        for i, (_, scaler) in enumerate(scalers.items()):
            ypred[:, i:i + 1] = scaler.inverse_transform(ypred[:, i:i + 1])
            ytrue[:, i:i + 1] = scaler.inverse_transform(ytrue[:, i:i + 1])

    if threshold is not None:
        upper, lower = threshold
        mask = np.ones(shape=ytrue.shape, dtype=bool)
        if upper < lower:
            upper, lower = lower, upper
        for i, x in enumerate(ytrue):
            if upper > x > lower:
                mask[i] = False

        ypred = ypred[mask]
        ytrue = ytrue[mask]

    result = np.sqrt(np.mean(np.square(ypred - ytrue)))

    if np.isnan(result):
        return np.inf

    return result


def mae(y_true, y_pred, scalers=None, threshold=None):
    if len(y_pred) == 0 or len(y_true) == 0:
        return np.inf
    ypred = y_pred.copy()
    ytrue = y_true.copy()

    if scalers is not None:
        for i, (_, scaler) in enumerate(scalers.items()):
            ypred[:, i:i + 1] = scaler.inverse_transform(ypred[:, i:i + 1])
            ytrue[:, i:i + 1] = scaler.inverse_transform(ytrue[:, i:i + 1])

    if threshold is not None:
        upper, lower = threshold
        mask = np.ones(shape=ytrue.shape, dtype=bool)
        if upper < lower:
            upper, lower = lower, upper
        for i, x in enumerate(ytrue):
            if upper > x > lower:
                mask[i] = False
        ypred = ypred[mask]
        ytrue = ytrue[mask]

    result = np.mean(np.abs(ypred - ytrue))

    if np.isnan(result):
        return np.inf

    return result


def extract_device_predictions(dataframe, devicestatus, prediction_horizon):
    valid_status = [status for status in devicestatus
                    if dataframe.index[-1] > status['date'] + timedelta(minutes=5) > dataframe.index[0]
                    and 'predBGs' in status.keys()]

    for status in valid_status:
        raise NotImplementedError("Versuche einen Weg zu finden die Predictions auf die rolled DataFrames abzubilden")


def closest_date(date=None, dates=None, return_index=False, return_dataframe=False):
    if not isinstance(dates, DataFrame):
        df = DataFrame(dates, columns=['date'])
        df['datetime'] = pd.to_datetime(df['date'])
        df = df.set_index('datetime')
        df.drop(['date'], axis=1, inplace=True)
        if return_dataframe:
            return df
        cd_index = df.index.get_loc(date, method='nearest')
        if return_index:
            return dates[cd_index], cd_index
        return dates[cd_index]
    else:
        cd_index = dates.index.get_loc(date, method='nearest')
        if return_index:
            return dates.index[cd_index], cd_index
        return dates.index[cd_index]


def plot_boxplot(preds, tools, width=560, height=300, scalers=None, metric='MAE'):

    cats = list(preds.keys())
    found = False
    boxes = list()
    for label, prediction in preds.items():
        ypred = prediction['pred']
        ytrue = prediction['y'][:len(ypred)]
        mask = ~np.isnan(ypred)
        ypred = ypred[mask].reshape((np.count_nonzero(mask), ypred.shape[-1]))
        ytrue = ytrue[mask].reshape((np.count_nonzero(mask), ypred.shape[-1]))
        score = np.empty(ypred.shape)
        score[:] = 100000
        for i in range(len(ypred)):
            if metric == 'MAE':
                error = mae(ytrue[i:i+1], ypred[i:i+1], scalers=scalers)
            elif metric == "RMSE":
                error = rmse(ytrue[i:i + 1], ypred[i:i + 1], scalers=scalers)
            elif 'RMSE' in metric and "(" in metric:
                error = rmse(ytrue[i:i + 1], ypred[i:i + 1], scalers=scalers, threshold=(180, 70))
            elif 'MAE' in metric and "(" in metric:
                error = rmse(ytrue[i:i + 1], ypred[i:i + 1], scalers=scalers, threshold=(180, 70))
            else:
                raise Exception("Metric {} not understood".format(metric))
            score[i:i+1] = error

        if len(score) == 0:
            continue

        score = score.flatten()
        df = pd.DataFrame(dict(group=[label for _ in score], score=score))
        boxes.append((label, df))
        found = True

    p = figure(tools=tools, width=width, height=height, x_range=[label for label, _ in boxes])
    p.output_backend = "svg"
    for label, df in boxes:
        groups = df.groupby('group')
        q1 = groups.quantile(q=0.25)
        q2 = groups.quantile(q=0.5)
        q3 = groups.quantile(q=0.75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        lower = q1 - 1.5 * iqr

        def outliers(group):
            category = group.name
            return group[(group.score > upper.loc[category][0]) | (group.score < lower.loc[category][0])]['score']

        out = groups.apply(outliers).dropna()

        outx = list()
        outy = list()

        if label in out.index:
            for value in out.loc[label]:
                outx.append(label)
                outy.append(value)

        qmin = groups.quantile(q=0.00)
        qmax = groups.quantile(q=1.00)
        upper.score = [min([x, y]) for (x, y) in zip(list(qmax.iloc[:, 0]), upper.score)]
        lower.score = [max([x, y]) for (x, y) in zip(list(qmin.iloc[:, 0]), lower.score)]

        p.segment([label], upper.score, [label], q3.score, line_width=2, line_color='black')
        p.segment([label], lower.score, [label], q1.score, line_width=2, line_color='black')

        p.rect([label], (q3.score + q2.score) / 2, 0.7, q3.score - q2.score,
               fill_color='#E08E79', line_width=2, line_color='black')
        p.rect([label], (q2.score + q1.score) / 2, 0.7, q2.score - q1.score,
               fill_color='#E08E79', line_width=2, line_color='black')

        p.rect([label], lower.score, 0.2, 0.01, line_color='black')
        p.rect([label], upper.score, 0.2, 0.01, line_color='black')

        p.circle(outx, outy, size=6, color='#F38630', fill_alpha=0.6)

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = 'white'
    p.grid.grid_line_width = 2
    p.xaxis.major_label_text_font_size = '6pt'
    p.yaxis.axis_label = metric
    p.xaxis.axis_label = "Models"

    if found:
        return [Panel(child=p, title='BoxPlots')]
    else:
        return list()

def get_from_dic(dic, i):
    return dic[list(dic.keys())[i]]
