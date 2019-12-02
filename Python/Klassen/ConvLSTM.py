import warnings

from helperfunctions import rmse, rmse_weighted, root_mean_squared_error, root_mean_squared_error_weighted, mae
import keras
import json
import numpy as np

from keras import Sequential
from keras.layers import Dense, ConvLSTM2D, TimeDistributed, BatchNormalization, MaxPooling3D, Flatten


class DataPrep(keras.utils.Sequence):
    def __init__(self, x, y, batch_size, frames, shuffle=False, show_warnings=False):
        self.batch_size = int(batch_size / frames)
        if self.batch_size == 0:
            if show_warnings:
                warnings.warn("Batch Size is 0 setting it to 1")
            self.batch_size = 1
        remainder = int(x.shape[0] / frames)

        # (samples, frames, channels, rows, cols)
        self.x = x[:remainder * frames].reshape(remainder, frames, 1, x.shape[1], x.shape[2])
        # (samples, frames, label)
        self.y = y[:remainder * frames].reshape(remainder, frames, y.shape[1])
        self.shuffle = shuffle

    def __getitem__(self, index):
        # print("Index:", index)
        return self.x[index * self.batch_size:(index + 1) * self.batch_size], \
               self.y[index * self.batch_size:(index + 1) * self.batch_size]

    def __len__(self):
        'Denotes the number of batches per epoch'
        # print("Len:", int(np.floor(len(self.x) / self.batch_size)))
        return int(np.floor(len(self.x) / self.batch_size))

    def __data_generation(self, **kwargs):
        print("kwargs", kwargs)

    def on_epoch_end(self):
        if self.shuffle:
            p = np.random.permutation(len(self.x))
            self.x = self.x[p]
            self.y = self.y[p]


class ConvLSTM:
    def __init__(self, config_file, print_summary=False, show_warnings=False):
        self.configs = json.load(open(config_file, 'r'))
        self.model = None
        self.model = self.build_model('training', print_summary, show_warnings)
        self.pred_model = None
        self.eval_model = None

    def build_model(self, mode, print_summary=False, show_warnings=False):
        scope = self.configs[mode]

        frames = scope['frames'] if 'frames' in scope else None
        batch_size = int(scope['batch_size'] / frames) if 'batch_size' in scope else 32
        if batch_size == 0:
            if show_warnings:
                warnings.warn("Bach_size is 0 setting it to 1op")
            batch_size = 1
        n_features = scope['n_features'] if 'n_features' else 1

        model = Sequential()
        for layer in self.configs['model']['layers']:
            kwargs = {}
            layer_type = layer['type'].lower() if 'type' in layer else None
            filters = layer['filters'] if 'filters' in layer else None
            kernel_size = (layer['kernel_size'],) + (1,) if 'kernel_size' in layer else None
            kernel_initializer = layer['kernel_initializer'] if 'kernel_initializer' in layer else 'glorot_uniform'
            data_format = layer['data_format'] if 'data_format' in layer else 'channels_first'
            timesteps = layer['timesteps'] if 'timesteps' in layer else None
            channels = layer['channels'] if 'channels' in layer else 1
            padding = layer['padding'] if 'padding' in layer else 'same'
            return_sequences = layer['return_sequences'] if 'return_sequences' in layer else False
            batch_input_shape = (batch_size, frames, channels, timesteps, n_features) if timesteps is not None else None
            pool_size = (1,) + (layer['pool_size'],) + (1,) if 'pool_size' in layer else None
            stateful = layer['stateful'] if 'stateful' in layer else False
            output_dimension = layer['output_dimension'] if 'output_dimension' in layer else None

            if batch_input_shape is not None:
                kwargs['batch_input_shape'] = batch_input_shape

            if layer_type == 'convlstm2d':
                model.add(ConvLSTM2D(filters=filters,
                                     kernel_size=kernel_size,
                                     kernel_initializer=kernel_initializer,
                                     data_format=data_format,
                                     padding=padding,
                                     return_sequences=return_sequences,
                                     stateful=stateful,
                                     **kwargs))
            elif layer_type == 'batchnormalization':
                model.add(BatchNormalization())
            elif layer_type == 'maxpooling3d':
                model.add(MaxPooling3D(pool_size=pool_size,
                                       padding=padding,
                                       data_format=data_format))
            elif layer_type == 'timedistributed_flatten':
                model.add(TimeDistributed(Flatten()))
            elif layer_type == 'timedistributed_dense':
                model.add(TimeDistributed(Dense(output_dimension)))

        loss = self.configs['model']['loss']
        if loss.lower() == 'rmse':
            loss = root_mean_squared_error
        elif loss.lower() == 'rmse_weighted':
            loss = root_mean_squared_error_weighted

        optimizer = self.configs['model']['optimizer']

        model.compile(loss=loss, optimizer=optimizer)
        if self.model is not None:
            model.set_weights(self.model.get_weights())

        if print_summary:
            model.summary()

        return model

    def fit(self, x=None, y=None, validation_data=None, dic=None, val_dic=None,
            shuffle=False,
            epochs=None, callbacks=None, **kwargs):
        scope = self.configs['training']
        batch_size = scope['batch_size']
        frames = scope['frames']
        if epochs is None:
            epochs = scope['epochs'] if 'epochs' in scope else 35

        if validation_data is not None:
            val_x, val_y = validation_data
        else:
            val_x, val_y = None, None

        if dic is not None:
            x, y = None, None
            for _id, indexes in dic.items():
                for index, data in indexes.items():
                    if x is None or y is None:
                        x = np.empty(((0,) + data['x'].shape[1:]))
                        y = np.empty(((0,) + data['y'].shape[1:]))

                    x = np.append(x, data['x'], axis=0)
                    y = np.append(y, data['y'], axis=0)

        if val_dic is not None:
            val_x, val_y = list(), list()
            for _id, indexes in val_dic.items():
                for index, data in indexes.items():
                    val_x.extend(data['x'].tolist())
                    val_y.extend(data['y'].tolist())
            val_x = np.array(val_x)
            val_y = np.array(val_y)

        fit_gen = DataPrep(x, y, batch_size=batch_size, frames=frames, shuffle=shuffle)
        validation_steps = None

        val_gen = None
        if val_x is not None and val_y is not None:
            val_gen = DataPrep(val_x, val_y, batch_size=batch_size, frames=frames, shuffle=shuffle)
            validation_steps = len(val_gen)
        history = self.model.fit_generator(generator=fit_gen, validation_data=val_gen,
                                           steps_per_epoch=len(fit_gen), validation_steps=validation_steps,
                                           epochs=epochs,
                                           callbacks=callbacks)
        return history

    def predict(self, x, y, dic=None, model=None, verbose=0, **kwargs):
        scope = self.configs['predicting']
        batch_size = scope['batch_size'] if 'batch_size' in scope else self.model.input_shape[0]
        frames = scope['frames'] if 'frames' in scope else self.model.input_shape[1]
        if self.pred_model is None:
            if batch_size == self.model.input_shape[0] and frames == self.model.input_shape[1]:
                self.pred_model = self.model
            else:
                self.pred_model = self.build_model('predicting')

        model = model if model is not None else self.pred_model

        if dic is not None:
            predictions = {}
            count = 0
            for _id, windows in dic.items():
                for _, data in windows.items():
                    x_, y_ = np.array(data['x']), np.array(data['y'])
                    pred_gen = DataPrep(x=x_, y=y_, batch_size=batch_size, frames=frames)
                    pred = model.predict_generator(pred_gen, len(pred_gen))
                    pred = pred.reshape((pred.shape[0] * pred.shape[1], pred.shape[-1]))
                    predictions[count] = {'id': _id,
                                          'x': x_,
                                          'y': y_,
                                          'pred': pred}
                    count += 1
            return predictions
        else:
            pred_gen = DataPrep(x=x, y=y, batch_size=batch_size, frames=frames)
            predictions = {}
            preds = model.predict_generator(pred_gen, steps=len(pred_gen))
            count = 0
            for i in range(batch_size, len(preds) + 1, batch_size):
                predictions[count] = {'x': x[i - batch_size:i],
                                      'y': y[i - batch_size:i],
                                      'pred': preds[i - batch_size:i]}
                count += 1
            return predictions

    def evaluate(self, x, y, dic=None, scaler=None, verbose=0, **kwargs):
        scope = self.configs['evaluating']
        batch_size = scope['batch_size'] if 'batch_size' in scope else self.model.input_shape[0]
        frames = scope['frames'] if 'frames' in scope else self.model.input_shape[1]
        if self.eval_model is None:
            if batch_size == self.model.input_shape[0] and frames == self.model.input_shape[1]:
                self.eval_model = self.model
            else:
                self.eval_model = self.build_model('evaluating')
        output_shape = scope['output_shape'] if 'output_shape' in scope else (1,)

        predictions = self.predict(x, y, dic, model=self.eval_model, verbose=verbose, **kwargs)
        error = {}
        for i, prediction in predictions.items():
            ypred = prediction['pred']
            ytrue = prediction['y'][:len(ypred)]
            mask = ~np.isnan(ypred)
            ypred = ypred[mask].reshape((np.count_nonzero(mask),) + output_shape)
            ytrue = ytrue[mask].reshape((np.count_nonzero(mask),) + output_shape)
            error[i] = {'RMSE_{}'.format(len(ytrue)): rmse(ypred, ytrue, scaler),
                        'RMSE_{}(x>180x<70)'.format(len(ytrue)): rmse(ypred, ytrue, scaler, (180, 70)),
                        'MAE_{}'.format(len(ytrue)): mae(ypred, ytrue, scaler),
                        'MAE_{}(180>x>70)'.format(len(ytrue)): mae(ypred, ytrue, scaler, (180, 70))}

        return predictions, error

    def reset(self):
        self.model = None
        self.model = self.build_model('training')
        self.pred_model = None
        self.eval_model = None


    def export_configs(self, label, path):
        with open('/'.join([path, label]) + ".json", 'w') as jsonfile:
            json.dump(self.configs, jsonfile)
