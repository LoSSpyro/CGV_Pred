import numpy as np
import keras
import json

from keras import Sequential
from keras.layers import LSTM, Dense, Bidirectional, BatchNormalization

from helperfunctions import root_mean_squared_error, root_mean_squared_error_weighted, rmse, mae


class DataPrep(keras.utils.Sequence):
    def __init__(self, x, y, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.x, self.y = x, y
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


class BaseLSTM:
    def __init__(self, config_file, print_summary=False):
        self.configs = json.load(open(config_file, 'r'))
        self.model = None
        self.model = self.build_model('training', print_summary=print_summary)
        self.pred_model = None
        self.eval_model = None

    def build_model(self, mode, print_summary=False):
        scope = self.configs[mode]
        batch_size = scope['batch_size'] if 'batch_size' in scope else 32
        n_features = scope['n_features'] if 'n_features' in scope else 1

        model = Sequential()
        for layer in self.configs['model']['layers']:
            kwargs = {}
            layer_type = layer['type'].lower() if 'type' in layer else None
            neurons = layer['neurons'] if 'neurons' in layer else 1
            timesteps = layer['timesteps'] if 'timesteps' in layer else None
            return_sequences = layer['return_sequences'] if 'return_sequences' in layer else True
            stateful = layer['stateful'] if 'stateful' in layer else False
            dropout = layer['dropout'] if 'dropout' in layer else 0.0
            batch_input_shape = (batch_size, timesteps, n_features) if timesteps is not None else None
            kernel_regularizer = layer['kernel_regularizer'] if 'kernel_regularizer' in layer else None
            kr_value = layer['kr_value'] if 'kr_value' in layer else 0.01
            batch_normalization = layer['batch_normalization'] if 'batch_normalization' in layer else False

            if batch_input_shape is not None:
                kwargs['batch_input_shape'] = batch_input_shape

            if kernel_regularizer == "l1":
                kernel_regularizer = keras.regularizers.l1(kr_value)
            elif kernel_regularizer == "l2":
                kernel_regularizer = keras.regularizers.l2(kr_value)

            if layer_type == 'lstm':
                activation = layer['activation'] if 'activation' in layer else 'tanh'
                recurrent_activation = layer['recurrent_activation'] if 'recurrent_activation' in layer else 'sigmoid'
                model.add(LSTM(neurons,
                               activation=activation, recurrent_activation=recurrent_activation,
                               return_sequences=return_sequences,
                               stateful=stateful,
                               dropout=dropout,
                               kernel_regularizer=kernel_regularizer,
                               **kwargs))
            if layer_type == 'bidirectional':
                activation = layer['activation'] if 'activation' in layer else 'tanh'
                recurrent_activation = layer['recurrent_activation'] if 'recurrent_activation' in layer else 'sigmoid'
                model.add(Bidirectional(LSTM(neurons,
                                             activation=activation, recurrent_activation=recurrent_activation,
                                             return_sequences=return_sequences,
                                             stateful=stateful,
                                             dropout=dropout),
                                        **kwargs))
            if layer_type == 'dense':
                model.add(Dense(neurons, **kwargs))
            if batch_normalization:
                model.add(BatchNormalization())

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

    def fit(self, x=None, y=None, dic=None, validation_data=None, val_dic=None, epochs=None, callbacks=None, shuffle=False, **kwargs):
        scope = self.configs['training']
        batch_size = scope['batch_size'] if 'batch_size' in scope else 32
        if epochs is None:
            epochs = scope['epochs'] if 'epochs' in scope else 100
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

        train_gen = DataPrep(x, y, batch_size=batch_size, shuffle=shuffle)
        val_gen = None
        validation_steps = None
        if val_x is not None and val_y is not None:
            val_gen = DataPrep(val_x, val_y, batch_size=batch_size, shuffle=shuffle)
            validation_steps = len(val_gen)
        history = self.model.fit_generator(generator=train_gen, validation_data=val_gen,
                                           steps_per_epoch=len(train_gen), validation_steps=validation_steps,
                                           epochs=epochs,
                                           callbacks=callbacks)
        return history

    def predict(self, x, y, dic=None, model=None, verbose=0, **kwargs):
        scope = self.configs['predicting']
        batch_size = scope['batch_size'] if 'batch_size' in scope else self.model.input_shape[0]
        if self.pred_model is None:
            if batch_size == self.model.input_shape[0]:
                self.pred_model = self.model
            else:
                self.pred_model = self.build_model('predicting')

        model = model if model is not None else self.pred_model

        if dic is not None:
            predictions = {}
            i = 0
            for _id, frames in dic.items():
                for _, data in frames.items():
                    x_, y_ = np.array(data['x']), np.array(data['y'])
                    pred_gen = DataPrep(x=x_, y=y_, batch_size=batch_size)
                    predictions[i] = {'id': _id,
                                      'x': x_,
                                      'y': y_,
                                      'pred': model.predict_generator(pred_gen, len(pred_gen))}
                    i += 1
            return predictions
        else:
            pred_gen = DataPrep(x=x, y=y, batch_size=batch_size)
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
        if self.eval_model is None:
            if batch_size == self.model.input_shape[0]:
                self.eval_model = self.model
            else:
                self.eval_model = self.build_model('evaluating')
        output_shape = scope['output_shape'] if 'output_shape' in scope else (1,)

        predictions = self.predict(x, y, dic, model=self.eval_model, verbose=verbose, **kwargs)
        error = {}
        for i, prediction in predictions.items():
            try:
                ypred = prediction['pred']
                ytrue = prediction['y']
                if len(ypred) == 0:
                    ypred = np.ones(ytrue.shape)
                    ypred[:] = np.NaN
                ytrue = ytrue[:len(ypred)]
                mask = ~np.isnan(ypred)
                ypred = ypred[mask].reshape((np.count_nonzero(mask),) + output_shape)
                ytrue = ytrue[mask].reshape((np.count_nonzero(mask),) + output_shape)
                error[i] = {'RMSE_{}'.format(len(ytrue)): rmse(ypred, ytrue, scaler),
                            'RMSE_{}(x>180x<70)'.format(len(ytrue)): rmse(ypred, ytrue, scaler, (180, 70)),
                            'MAE_{}'.format(len(ytrue)): mae(ypred, ytrue, scaler),
                            'MAE_{}(180>x>70)'.format(len(ytrue)): mae(ypred, ytrue, scaler, (180, 70))}
            except Exception as e:
                print(i)
                print(type(prediction))
                print(prediction)
                raise e

        return predictions, error

    def reset(self):
        self.model = None
        self.model = self.build_model('training')
        self.pred_model = None
        self.eval_model = None

    def export_configs(self, label, path):
        with open('/'.join([path, label]) + ".json", 'w') as jsonfile:
            json.dump(self.configs, jsonfile)
