from _warnings import warn
from collections import defaultdict

from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy.ma as ma

from CGEGA.cg_ega.cg_ega import CG_EGA
from helperfunctions import rmse, mae, plot_boxplot
from bokeh.models.widgets import Div
from bokeh.models import Panel, Tabs, HoverTool, ColumnDataSource, Range1d, LinearAxis
from bokeh.layouts import row, column
from bokeh.plotting import figure
from bokeh.io import show, reset_output, output_notebook
from bokeh.palettes import Category10_10 as palette
import numpy as np
import pandas as pd
import os


class ModelContainer:
    def __init__(self, model_container=None, model_list=None, fitted=False, predicted=False, not_kwargs=None):
        if model_container is not None:
            self.model_dict = model_container.model_dict if 'model_dict' in vars(model_container).keys() else {}
            self.fitted = model_container.fitted if 'fitted' in vars(model_container).keys() else defaultdict(
                lambda: fitted)
            self.evaluated = model_container.evaluated if 'evaluated' in vars(model_container).keys() else defaultdict(
                lambda: False)
            self.predicted = model_container.predicted if 'predicted' in vars(model_container).keys() else defaultdict(
                lambda: predicted)
            self.loss = model_container.loss if 'loss' in vars(model_container).keys() else {}
            self.predictions = model_container.predictions if 'predictions' in vars(model_container).keys() else {}
            self.not_kwargs = model_container.not_kwargs if 'not_kwargs' in vars(
                model_container).keys() else defaultdict(list)
            self.errors = model_container.errors if 'errors' in vars(model_container).keys() else {}
            self.scores = model_container.scores if 'scores' in vars(model_container).keys() else {}
            self.cg_egas = model_container.cg_egas if 'cg_egas' in vars(model_container).keys() else {}
            self.mean_cg_egas = model_container.mean_cg_egas if 'mean_cg_egas' in vars(model_container).keys() else {}
        else:
            self.model_dict = model_list if model_list is not None else {}
            self.fitted = defaultdict(lambda: fitted)
            self.evaluated = defaultdict(lambda: False)
            self.predicted = defaultdict(lambda: predicted)
            self.loss = {}
            self.predictions = {}
            self.not_kwargs = not_kwargs if not_kwargs is not None else defaultdict(list)
            self.errors = {}
            self.scores = {}
            self.cg_egas = {}
            self.mean_cg_egas = {}

    def add_model(self, model, label, not_kwargs=None, override=False, show_warnings=False):
        if label in self.model_dict.keys():
            if show_warnings:
                warn("Label {} already in Container".format(label))
            if override:
                self.model_dict[label] = model
                self.fitted[label] = False
                self.evaluated[label] = False
                self.predicted[label] = False
                self.not_kwargs[label] = not_kwargs if not_kwargs is not None else list()
                if show_warnings:
                    warn("Overriding {}".format(label))
            elif show_warnings:
                warn("Model not saved")
        else:
            self.model_dict[label] = model
            self.not_kwargs[label] = not_kwargs if not_kwargs is not None else list()

    def remove_model(self, label):

        if label in self.model_dict.keys():
            self.model_dict.pop(label)
        if label in self.fitted.keys():
            self.fitted.pop(label)
        if label in self.evaluated.keys():
            self.evaluated.pop(label)
        if label in self.predicted.keys():
            self.predicted.pop(label)
        if label in self.not_kwargs.keys():
            self.not_kwargs.pop(label)
        if label in self.errors.keys():
            self.errors.pop(label)
        if label in self.scores.keys():
            self.scores.pop(label)
        if label in self.cg_egas.keys():
            self.cg_egas.pop(label)
        if label in self.mean_cg_egas.keys():
            self.mean_cg_egas.pop(label)

    def reset_model(self, label):
        if label not in self.model_dict.keys():
            raise Exception("{} Model not Found".format(label))
        self.model_dict[label].reset()
        self.fitted[label] = False
        self.evaluated[label] = False
        self.predicted[label] = False

    def reset_models(self):
        for label, history in self.fitted.items():
            if history is not None:
                self.reset_model(label)

    def load_model(self, label, path=""):
        try:
            self.model_dict[label].model.load_weights(filepath=path + label)
            self.fitted[label] = True
        except Exception as e:
            print(path + label, "not found")

    def add_invalid_parameter(self, label, var):
        self.not_kwargs[label].append(var)

    def fit(self, x=None, y=None, w=None, dic=None, validation_data=None, val_dic=None,
            in_features=None, out_features=None,
            epochs=0, callbacks=None, monitor='val_loss', patience=5, min_delta=0.001, save_best_only=True,
            shuffle=False,
            path=None, refit=False,
            verbose=0, plot_val=False):

        if callbacks is None:
            callbacks = list()

        path = path + "/{}" if path is not None else 'model_weights/{}'
        if not os.path.isdir(path.split("/")[0]):
            os.mkdir(path.split("/")[0])

        for label, model in self.model_dict.items():
            if self.fitted[label] and not refit:
                continue

            not_kwargs = self.not_kwargs[label]

            x_train = x if 'x' not in not_kwargs else None
            y_train = y if 'y' not in not_kwargs else None
            w_train = w if 'w' not in not_kwargs else None
            val_data = validation_data if 'validation_data' not in not_kwargs else None
            in_feat = in_features if 'in_features' not in not_kwargs else ['sgv']
            out_feat = out_features if 'out_features' not in not_kwargs else ['sgv']
            n_epochs = epochs if 'epochs' not in not_kwargs else 0
            verb = verbose if 'verbose' not in not_kwargs else 0
            shuffl = shuffle if 'shuffle' not in not_kwargs else False
            plt_val = plot_val if 'plot_val' not in not_kwargs else False
            data = dic if 'dic' not in not_kwargs else None
            val_dic_data = val_dic if 'val_dic' not in not_kwargs else None
            cb = callbacks if 'callbacks' not in not_kwargs else list()
            cbs = []
            for callback in cb:
                if 'modelcheckpoint' == callback.lower():
                    cbs.append(ModelCheckpoint(filepath=path.format(label), monitor=monitor,
                                               save_best_only=save_best_only, verbose=1))
                elif 'earlystopping' == callback.lower():
                    cbs.append(EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience))
            if len(cbs) == 0:
                cbs = None

            if verb == 1:
                print("Fit", label)

            self.loss[label] = model.fit(x=x_train, y=y_train, w=w_train, dic=data, validation_data=val_data,
                                         val_dic=val_dic_data,
                                         in_features=in_feat, out_features=out_feat,
                                         epochs=n_epochs, callbacks=cbs, shuffle=shuffl, plot_val=plot_val)
            self.fitted[label] = True

        return self.loss

    def predict(self, x=None, y=None, dic=None,
                in_features=None, out_features=None,
                steps=1, verbose=0, scaler=None,
                ignore_models=None, repredict=False):
        ignore_models = ignore_models if ignore_models is not None else list()

        for label, model in self.model_dict.items():
            if self.predicted[label] and not repredict:
                continue

            not_kwargs = self.not_kwargs[label]

            x_test = x if 'x' not in not_kwargs else None
            y_test = y if 'y' not in not_kwargs else None
            in_feat = in_features if 'in_features' not in not_kwargs else ['sgv']
            out_feat = out_features if 'out_features' not in not_kwargs else ['sgv']
            verb = verbose if 'verbose' not in not_kwargs else 0
            stp = steps if 'steps' not in not_kwargs else 1
            scale = scaler if 'scaler' not in not_kwargs else None
            id_dict = dic if 'dic' not in not_kwargs else None

            if label in ignore_models:
                continue

            if verb == 1:
                print("Predict", label)

            predictions = model.predict(x=x_test, y=y_test, dic=id_dict,
                                        in_features=in_feat, out_features=out_feat,
                                        steps=stp, scaler=scale, verbose=verb)
            if isinstance(predictions, dict) and any(['openaps' in label.lower() for label in predictions.keys()
                                                      if isinstance(label, str)]):
                for key, value in predictions.items():
                    lab = "_".join([label, key])

                    self.predictions[lab] = value
            else:
                self.predictions[label] = predictions
            self.predicted[label] = True

        return self.predictions

    def evaluate(self, x=None, y=None, dic=None,
                 in_features=None, out_features=None,
                 steps=1,
                 scaler=None,
                 verbose=0,
                 reeval=False, override_preds=True):
        for label, model in self.model_dict.items():
            if self.evaluated[label] and not reeval:
                continue
            self.evaluated[label] = True
            if verbose == 1:
                print("Evaluating:", label)
            predictions, errors = model.evaluate(x=x, y=y, dic=dic, steps=steps,
                                                 in_features=in_features, out_features=out_features,
                                                 scaler=scaler, verbose=verbose)
            if override_preds:
                if isinstance(predictions, dict) and 'openaps' in label.lower():
                    for key, value in predictions.items():
                        lab = "_".join([label, key])

                        self.predictions[lab] = value
                else:
                    self.predictions[label] = predictions

            if isinstance(errors, dict) and 'openaps' in label.lower():
                for key, value in errors.items():
                    lab = "_".join([label, key])

                    self.errors[lab] = value
                    self.evaluated[lab] = True
            else:
                self.errors[label] = errors

    def plot_data(self, dic, n=np.inf,
                  width=560, height=300,
                  tools=None,
                  features=None, scalers=None,
                  share_ranges=True):

        yrange = (-1, 1) if scalers is None else (20, 400)
        y_label = "Bloodsugar [mg/dL]" if scalers is not None else "Bloodsugar (scaled) [mg/dL]"
        standard = "pan,wheel_zoom,box_zoom,reset,save"
        tools = ','.join([standard, tools]) if tools is not None else standard

        reset_output()
        output_notebook()

        x_range, y_range = None, None
        for _id, frames in dic.items():
            tabs = list()
            for i, data in frames.items():
                if i > n:
                    break

                y = data['y'].copy()
                x_length = len(y)
                x = [g * 5 for g in range(len(y))]
                x_input = data['x'].copy()[:x_length]

                if scalers is not None:
                    for j, (_, scaler) in enumerate(scalers.items()):
                        for k in range(len(x_input)):
                            x_input[k, :, j:j + 1] = scaler.inverse_transform(x_input[k, :, j:j + 1])
                        if j == 0:
                            y[:, j:j + 1] = scaler.inverse_transform(y[:, j:j + 1])

                df = pd.DataFrame()
                df['x_index'] = x
                df['y'] = y
                df['x_input'] = x_input[:, -1, 0]

                if features is not None:
                    maxes = list()
                    for j, feature in enumerate(features):
                        value = x_input[:x_length, -1, j + 1:j + 2]
                        df[feature + 'top'] = value
                        maxes.append(value)
                    df['left'] = [g - 2.5 for g in x]
                    df['right'] = [g + 2.5 for g in x]

                df = ColumnDataSource(data=df)

                if share_ranges:
                    if x_range is None:
                        s = figure(width=width, height=height,
                                   x_range=(x[0] - 2.5, x[-1] + 10), y_range=yrange,
                                   x_axis_label="Time [minutes]", y_axis_label=y_label,
                                   tools=tools)
                        renderer = s.line(x='x_index', y='y', source=df, legend='Output', color=palette[0])
                        x_range, y_range = s.x_range, s.y_range
                    else:
                        s = figure(width=width, height=height, x_range=x_range, y_range=y_range,
                                   x_axis_label="Time [minutes]", y_axis_label=y_label, tools=tools)
                        renderer = s.line(x='x_index', y='y', source=df, legend='Output')
                else:
                    s = figure(width=width, height=height, y_range=yrange, x_axis_label="Time [minutes]", y_axis_label=y_label, tools=tools)
                    renderer = s.line(x='x_index', y='ypred', source=df, legend='Output')

                s.line(x='x_index', y='x_input', source=df, color=palette[2], legend="Last Value of Input")

                if features is not None:
                    s.extra_y_ranges = {'hist': Range1d(start=0, end=np.max(maxes) + 4)}
                    s.add_layout(LinearAxis(y_range_name='hist', axis_label="Insulin [IU]/Carbs [g]"), 'right')
                    for k, feature in enumerate(features):
                        s.quad(source=df, bottom=0, top=feature + 'top', left='left', right='right',
                               color=palette[k + 3], alpha=0.5, legend=feature,
                               y_range_name='hist')

                s.legend.click_policy = 'hide'
                s.output_backend = 'svg'

                txt = "ID: " + _id + "</br>"
                p = Div(text=txt, width=width, style={'font-size': '200%'})

                tab = Panel(child=column(p, s), title=str(i))
                tabs.append(tab)
            ts = Tabs(tabs=tabs)
            show(ts)

    def plot(self, label=None, contains=True,
             n=None, xrange=None,
             scaler=None,
             show_input=False, show_true=True,
             width=560, height=300, share_ranges=True,
             tools=None,
             sort=True, **kwargs):

        if label is None:
            labels = list(self.errors.keys())
        else:
            labels = [label] if not isinstance(label, list) else label
        if contains:
            errors = dict(
                (k, self.errors[k]) for k in self.errors.keys() if any([l.lower() in k.lower() for l in labels]))
        else:
            errors = dict((k, self.errors[k]) for k in labels)
        self.plot_all(n=n, errors=errors,
                      xrange=xrange,
                      show_input=show_input, show_true=show_true,
                      width=width, height=height,
                      tools=tools, share_ranges=share_ranges,
                      sort=sort, scaler=scaler,
                      **kwargs)

    def plot_all(self, n=None, errors=None, xrange=None,
                 show_input=False, show_true=True, features=None,
                 width=560, height=300,
                 tools=None, share_ranges=True,
                 sort=True,
                 boxplots=['OpenAPS'], metric='RMSE',
                 scaler=None,
                 **kwargs):
        reset_output()
        output_notebook()
        niter = n if n is not None else len(list(self.predictions.values())[0])
        iter_list = niter if isinstance(niter, list) or isinstance(niter, range) else range(niter)

        TOOLTIPS = [
            ("Predicted", "@ypred"),
            ("True", "@ytrue"),
            ("ERROR", "@error")
        ]

        standard = "pan,wheel_zoom,box_zoom,reset,save"
        tools = ','.join([standard, tools]) if tools is not None else standard

        errors = errors if errors is not None else self.errors

        y_label = "Bloodsugar [mg/dL]" if scaler is not None else "Bloodsugar (scaled) [mg/dL]"

        for i in iter_list:
            tabs = list()
            if sort:
                sorted_errors = sorted(errors.items(),
                                       key=lambda item: item[1][i][list(item[1][i].keys())[0]])
            else:
                sorted_errors = errors.items()

            box_preds = dict(
                (k, v[i]) for k, v in self.predictions.items() if
                k in boxplots or any(k.startswith(l) for l in boxplots))
            box_tab = plot_boxplot(box_preds, tools, width=width, height=height, scalers=scaler, metric=metric)
            tabs.extend(box_tab)

            x_range, y_range = None, None
            yrange = (-1, 1) if scaler is None else (20, 400)

            for label, _ in sorted_errors:
                if label in boxplots or any(label.startswith(l) for l in boxplots):
                    continue
                predictions = self.predictions[label]

                if xrange is not None:
                    x_length = xrange
                else:
                    if isinstance(predictions, dict):
                        x_length = len(predictions[i]['pred'])
                    else:
                        raise AttributeError("xrange must be given when prediction is not a dict")

                x = [g * 5 for g in range(x_length)]
                y = predictions[i]['pred'][:x_length].copy()

                if scaler is not None:
                    for j, (_, scal) in enumerate(scaler.items()):
                        if j == 0:
                            mask = ma.masked_invalid(y[:, j:j + 1]).mask
                            inv_transfrom = scal.inverse_transform(y[:, j:j + 1][~mask].reshape(-1, 1)).flatten()
                            y[:, j:j + 1][~mask] = inv_transfrom

                df = pd.DataFrame()
                df['x_index'] = x
                df['ypred'] = y
                if features is not None:
                    maxes = list()
                    for j, feature in enumerate(features):
                        value = predictions[i]['x'][:x_length, -1, j + 1:j + 2]
                        df[feature + 'top'] = value
                        maxes.append(value)
                    df['left'] = [g - 2.5 for g in x]
                    df['right'] = [g + 2.5 for g in x]
                if show_true:
                    ytrue = predictions[i]['y'][:x_length].copy()
                    if scaler is not None:
                        for j, (_, scal) in enumerate(scaler.items()):
                            ytrue[:, j:j + 1] = scal.inverse_transform(ytrue[:, j:j + 1])
                    df['ytrue'] = ytrue
                if show_input:
                    if 'x' in predictions[i].keys():
                        x_input = predictions[i]['x'][:x_length, -1, :1].copy()
                        if scaler is not None:
                            for j, (_, scal) in enumerate(scaler.items()):
                                x_input[:, j:j + 1] = scal.inverse_transform(x_input[:, j:j + 1])

                        df['input'] = x_input
                    else:
                        x_tmp = np.zeros(df['x_index'].shape)
                        if scaler is not None:
                            x_tmp[:, :, 0:1] = (yrange[0] + yrange[1]) / 2.0
                        df['input'] = x_tmp

                df['error'] = abs(df['ypred'] - df['ytrue'])
                df = ColumnDataSource(data=df)

                if share_ranges:
                    if x_range is None:
                        s = figure(width=width, height=height, y_range=yrange,
                                   x_axis_label="Time [minutes]", y_axis_label=y_label,
                                   tools=tools)
                        renderer = s.line(x='x_index', y='ypred', source=df, legend='y_pred', color=palette[0])
                        x_range, y_range = s.x_range, s.y_range
                    else:
                        s = figure(width=width, height=height, x_range=x_range, y_range=y_range,
                                   x_axis_label="Time [minutes]", y_axis_label=y_label,
                                   tools=tools)
                        renderer = s.line(x='x_index', y='ypred', source=df, legend='y_pred')
                else:
                    s = figure(width=width, height=height, y_range=(-1, 1),
                               x_axis_label="Time [minutes]", y_axis_label=y_label,
                               tools=tools)
                    renderer = s.line(x='x_index', y='ypred', source=df, legend='y_pred')

                if show_true:
                    s.line(x='x_index', y='ytrue', source=df, color=palette[1], legend='y_true')

                if show_input:
                    s.line(x='x_index', y='input', source=df, color=palette[2], legend='x_input')

                if features is not None:
                    s.extra_y_ranges = {'hist': Range1d(start=0, end=np.max(maxes) + 4)}
                    s.add_layout(LinearAxis(y_range_name='hist'), 'right')
                    for i, feature in enumerate(features):
                        s.quad(source=df, bottom=0, top=feature + 'top', left='left', right='right',
                               color=palette[i + 3], alpha=0.5, legend=feature,
                               y_range_name='hist')

                s.legend.click_policy = 'hide'

                if 'hover' in tools:
                    hover_tool = s.select(dict(type=HoverTool))
                    hover_tool.tooltips = TOOLTIPS
                    hover_tool.renderers = [renderer]
                    hover_tool.mode = 'vline'

                txt = label + "</br>"
                for metric, result in self.errors[label][i].items():
                    txt += metric + ": " + str(result) + "</br>"
                p = Div(text=txt, width=width, height=100)
                s.output_backend = 'svg'

                tab = Panel(child=row(s, p), title=label)
                tabs.append(tab)
            ts = Tabs(tabs=tabs)
            show(ts)

    def plot_loss(self, n=None, palett=None):
        palett = palett if palett is not None else palette
        for label, losses in self.loss.items():
            if losses is None:
                continue
            s = figure(title=label, width=560, height=350)
            for (lab, loss), color in zip(losses.history.items(), palett):
                s.line(list(range(len(loss))), loss, legend=lab, color=color)
            show(s)

    def score_models(self):
        score_keys = ('RMSE', 'RMSE(x>180x<70)', 'MAE', 'MAE(x>180x<70)')
        for label in self.errors.keys():
            self.scores[label] = {'best': {'RMSE': 0,
                                           'RMSE(x>180x<70)': 0,
                                           'MAE': 0,
                                           'MAE(x>180x<70)': 0},
                                  'mean': {'RMSE': 0,
                                           'RMSE(x>180x<70)': 0,
                                           'MAE': 0,
                                           'MAE(x>180x<70)': 0},
                                  'results': {'RMSE': 0,
                                              'RMSE(x>180x<70)': 0,
                                              'MAE': 0,
                                              'MAE(x>180x<70)': 0}}

        n = len(list(self.predictions.values())[0])
        print(n)
        for i in range(n):
            for m in range(4):
                sorted_errors = sorted(self.errors.items(),
                                       key=lambda item: item[1][i][list(item[1][i].keys())[m]])

                best = False
                for label, errors in sorted_errors:
                    error = errors[i]
                    if not np.isinf(error[list(error.keys())[m]]):
                        self.scores[label]['mean'][score_keys[m]] += error[list(error.keys())[m]]
                        self.scores[label]['results'][score_keys[m]] += 1
                    if not best:
                        self.scores[label]['best'][score_keys[m]] += 1
                        best = True

        for label, score in self.scores.items():
            for metric, value in score['mean'].items():
                if not score['results'][metric] == 0:
                    score['mean'][metric] = value / float(score['results'][metric])
                else:
                    score['mean'][metric] = np.inf

        return self.scores

    def do_cg_ega(self, frequence=5, scalers=None, redo=False):

        for model_label, prediction in self.predictions.items():
            if model_label not in self.cg_egas.keys() or redo:
                self.cg_egas[model_label] = {}
                for frameindex, data in prediction.items():
                    y_pred = data['pred'].copy()
                    y_true = data['y'].copy()[:len(y_pred)]
                    if len(y_pred) == 0:
                        print("Skip", model_label, frameindex)
                        continue
                    mask = ma.masked_invalid(y_pred).mask
                    if len(y_pred[~mask]) == 0:
                        continue
                    if scalers is not None:
                        ytrue_transform = scalers['sgv'].inverse_transform(y_true[~mask].reshape(-1, 1)).flatten()
                        ypred_transform = scalers['sgv'].inverse_transform(y_pred[~mask].reshape(-1, 1)).flatten()
                        y_true[~mask] = ytrue_transform
                        y_pred[~mask] = ypred_transform
                    if np.isnan(y_pred[-1]):
                        y_pred = y_pred[:-1]
                        y_true = y_true[:-1]
                    if np.isnan(y_pred[0]):
                        y_pred = y_pred[1:]
                        y_true = y_true[1:]
                    self.cg_egas[model_label][frameindex] = CG_EGA(y_true.reshape(1, -1), y_pred.reshape(1, -1),
                                                                   frequence)

    def evaluate_cg_ega(self, frequence=5, scalers=None, redo=False):
        if not self.cg_egas or redo:
            self.do_cg_ega(frequence, scalers, redo)
        for model_label, windows in self.cg_egas.items():
            self.mean_cg_egas[model_label] = {}
            n_ap_hypo, n_ap_euc, n_ap_hyper = 0, 0, 0
            n_be_hypo, n_be_euc, n_be_hyper = 0, 0, 0
            n_ep_hypo, n_ep_euc, n_ep_hyper = 0, 0, 0
            ap_hypo, ap_euc, ap_hyper = 0, 0, 0
            be_hypo, be_euc, be_hyper = 0, 0, 0
            ep_hypo, ep_euc, ep_hyper = 0, 0, 0
            for cgega in windows.values():
                tmp_ap_hypo, tmp_be_hypo, tmp_ep_hypo, tmp_ap_euc, tmp_be_euc, tmp_ep_euc, tmp_ap_hyper, tmp_be_hyper, tmp_ep_hyper = cgega.simplified()

                if not np.isnan(tmp_ap_hypo):
                    ap_hypo += tmp_ap_hypo
                    n_ap_hypo += 1

                if not np.isnan(tmp_ap_euc):
                    ap_euc += tmp_ap_euc
                    n_ap_euc += 1

                if not np.isnan(tmp_ap_hyper):
                    ap_hyper += tmp_ap_hyper
                    n_ap_hyper += 1

                if not np.isnan(tmp_be_hypo):
                    be_hypo += tmp_be_hypo
                    n_be_hypo += 1

                if not np.isnan(tmp_be_euc):
                    be_euc += tmp_be_euc
                    n_be_euc += 1

                if not np.isnan(tmp_be_hyper):
                    be_hyper += tmp_be_hyper
                    n_be_hyper += 1

                if not np.isnan(tmp_ep_hypo):
                    ep_hypo += tmp_ep_hypo
                    n_ep_hypo += 1

                if not np.isnan(tmp_ep_euc):
                    ep_euc += tmp_ep_euc
                    n_ep_euc += 1

                if not np.isnan(tmp_ep_hyper):
                    ep_hyper += tmp_ep_hyper
                    n_ep_hyper += 1

            mean_ap_hypo = ap_hypo / n_ap_hypo if n_ap_hypo > 0 else 0.00
            mean_be_hypo = be_hypo / n_be_hypo if n_be_hypo > 0 else 0.00
            mean_ep_hypo = ep_hypo / n_ep_hypo if n_ep_hypo > 0 else 0.00
            mean_ap_euc = ap_euc / n_ap_euc if n_ap_euc > 0 else 0.00
            mean_be_euc = be_euc / n_be_euc if n_be_euc > 0 else 0.00
            mean_ep_euc = ep_euc / n_ep_euc if n_ep_euc > 0 else 0.00
            mean_ap_hyper = ap_hyper / n_ap_hyper if n_ap_hyper > 0 else 0.00
            mean_be_hyper = be_hyper / n_be_hyper if n_be_hyper > 0 else 0.00
            mean_ep_hyper = ep_hyper / n_ep_hyper if n_ep_hyper > 0 else 0.00

            self.mean_cg_egas[model_label] = (mean_ap_hypo, mean_be_hypo, mean_ep_hypo,
                                              mean_ap_euc, mean_be_euc, mean_ep_euc,
                                              mean_ap_hyper, mean_be_hyper, mean_ep_hyper)

    def print_cg_egas(self, frequence=5, scalers=None, redo=False):
        if not self.mean_cg_egas or redo:
            self.evaluate_cg_ega(frequence, scalers, redo)
        for model, (
                ap_hypo, be_hypo, ep_hypo, ap_euc, be_euc, ep_euc, ap_hyper, be_hyper,
                ep_hyper) in self.mean_cg_egas.items():
            print(
                "\nModel: {}\nHypo\n\tAP: {}\n\tBE: {}\n\tEP: {}\nEuc\n\tAP: {}\n\tBE: {}\n\tEP: {}\nHyper\n\tAP: {}"
                "\n\tBE: {}\n\tEP: {}".format(
                    model, ap_hypo, be_hypo, ep_hypo, ap_euc, be_euc, ep_euc, ap_hyper, be_hyper, ep_hyper))

    def print_score(self, label=None, scores=None,
                    exact_labels=False,
                    kinds='best mean results', metrics='RMSE RMSE(x>180x<70) MAE MAE(x>180x<70  MAE(x>180x<70)',
                    rescore=False):
        if scores is None:
            if self.scores is None or rescore:
                self.score_models()
            scores = self.scores

        if label is None:
            label = list(scores.keys())
        if not isinstance(label, list):
            label = [label]

        sub_scores = {}
        if exact_labels:
            for label in label:
                sub_scores[label] = scores[label]
        else:
            for model_label, score in scores.items():
                if not any(l.lower() in model_label.lower() for l in label):
                    continue
                sub_scores[model_label] = score
        self.print_scores(scores=sub_scores, kinds=kinds, metrics=metrics)

    def print_scores(self, scores=None,
                     kinds='best mean results', metrics='RMSE RMSE(x>180x<70) MAE MAE(x>180x<70)',
                     rescore=False):
        if scores is None:
            if not self.scores or rescore:
                self.score_models()
            scores = self.scores

        kinds = kinds.split()
        metrics = metrics.split()

        for model, score in scores.items():
            print("\nModell:", model)
            for kind, metric_results in score.items():
                if kind not in kinds:
                    continue
                print(" {}".format(kind))
                for metric, result in metric_results.items():
                    if metric not in metrics:
                        continue
                    print("  {:<15}:{}".format(metric, result))

    def export_configs(self, path, labels=None):

        if not os.path.isdir(path):
            os.mkdir(path)

        if labels is None:
            labels = []
        if not isinstance(labels, list):
            labels = [labels]
        for label, model in self.model_dict.items():
            if hasattr(model, "export_configs") and label in labels:
                model.export_configs(label=label, path=path)

    def print_latex_scores(self, scores=None,
                           label=None, exact_labels=False, lti=None,
                           kinds='mean', metrics='RMSE RMSE(x>180x<70) MAE MAE(x>180x<70)',
                           rescore=False):
        if scores is None:
            if self.scores is None or rescore:
                self.score_models()
            scrs = self.scores

        if label is None:
            if scores is None:
                label = list(self.evaluated.keys())
            else:
                label = list(scrs.keys())
        if not isinstance(label, list):
            label = [label]

        sub_scores = {}
        if exact_labels:
            for label in label:
                sub_scores[label] = scrs[label]
        else:
            for model_label in label:
                if not any(l.lower() in model_label.lower() for l in label) or model_label == "OpenAPS":
                    continue
                score = scrs[model_label]
                sub_scores[model_label] = score

        for model, score in sub_scores.items():
            model_label = model if lti is None else lti[model]
            print("\n" + str(model_label), end="")
            for kind, metric_results in score.items():
                if kind not in kinds:
                    continue
                for metric, result in metric_results.items():
                    if metric not in metrics:
                        continue
                    if np.isinf(result):
                        print("&-", end="")
                    else:
                        print("&{0:3.2f}".format(result).replace('.', ','), end="")
            print("\\\\", end="")

    def print_latex_scores_cgega(self, mean_cg_egas=None,
                                 frequence=5, scalers=None, redo=False,
                                 label=None, exact_labels=False, lti=None):
        if mean_cg_egas is None:
            if self.mean_cg_egas is None or redo:
                self.evaluate_cg_ega(frequence=frequence, scalers=scalers, redo=redo)
            scrs = self.mean_cg_egas

        if label is None:
            if mean_cg_egas is None:
                label = list(self.evaluated.keys())
            else:
                label = list(scrs.keys())
        if not isinstance(label, list):
            label = [label]

        sub_scores = {}
        if exact_labels:
            for label in label:
                sub_scores[label] = scrs[label]
        else:
            for model_label in label:
                if not any(l.lower() in model_label.lower() for l in label) or model_label == "OpenAPS":
                    continue
                score = scrs[model_label]
                sub_scores[model_label] = score

        print(sub_scores.keys())
        for model, (ap_hypo, be_hypo, ep_hypo, ap_euc, be_euc, ep_euc, ap_hyper, be_hyper,
                    ep_hyper) in sub_scores.items():
            model_label = model if lti is None else lti[model]
            print("{model_label}&"
                  "{ap_hypo:0>2.{filler1}f}"
                  "&{be_hypo:0>2.{filler2}f}&"
                  "{ep_hypo:0>2.{filler3}f}&&"
                  "{ap_euc:0>2.{filler4}f}&"
                  "{be_euc:0>2.{filler5}f}&"
                  "{ep_euc:0>2.{filler6}f}&&"
                  "{ap_hyper:0>2.{filler7}f}&"
                  "{be_hyper:0>2.{filler8}f}&"
                  "{ep_hyper:0>2.{filler9}f}\\\\".format(
                    model_label=model_label,
                    ap_hypo=ap_hypo * 100, filler1=1 if ap_hypo >= 1 else 2,
                    be_hypo=be_hypo * 100, filler2=1 if be_hypo >= 1 else 2,
                    ep_hypo=ep_hypo * 100, filler3=1 if ep_hypo >= 1 else 2,
                    ap_euc=ap_euc * 100, filler4=1 if ap_euc >= 1 else 2,
                    be_euc=be_euc * 100, filler5=1 if be_euc >= 1 else 2,
                    ep_euc=ep_euc * 100, filler6=1 if ep_euc >= 1 else 2,
                    ap_hyper=ap_hyper * 100, filler7=1 if ap_hyper >= 1 else 2,
                    be_hyper=be_hyper * 100, filler8=1 if be_hyper >= 1 else 2,
                    ep_hyper=ep_hyper * 100, filler9=1 if ep_hyper >= 1 else 2
                    ).replace(".", ","))
