import warnings
from collections import defaultdict

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

from helperfunctions import scale_dataframe, convert_to_supervised, roll_dataframe


class Datahandler:
    def __init__(self, extract_samples=None, seed=42):
        self.__data_frames = None
        self.__entry_groups = None
        self.__scaled_dataframes = None
        self.__supervised_dataframes = None
        self.__rolled_dataframes = None
        self.scalers = None
        if extract_samples is not None and not callable(extract_samples):
            raise AttributeError("extract_samples has to be a function to extract samples from a data frame ("
                                 "Signature: extract_samples(dataframe, n_in, n_out)")
        self.__extract_samples = extract_samples
        np.random.seed(seed)

    @property
    def data_frames(self):
        if self.__data_frames is None:
            warnings.warn("No Dataframes set! Extracting DataFrames from the Datacontainer")
        else:
            return self.__data_frames

    @data_frames.setter
    def data_frames(self, value):
        self.__data_frames = value

    @property
    def entry_groups(self):
        if self.__entry_groups is None:
            warnings.warn("No Entry_groups set! Extracting them from Datacontainer.")

            self.__entry_groups = list()
            for identifier in self.__data.keys():
                datacontainer = self.__data[identifier]
                self.__entry_groups.extend(datacontainer.getEntryGroups(3 * 24 * 60 / 5, 30 * 60))

        return self.__entry_groups

    @entry_groups.setter
    def entry_groups(self, value):
        self.__entry_groups = value

    @property
    def scaled_dataframes(self):
        return self.__scaled_dataframes

    @property
    def rolled_dataframes(self):
        return self.__rolled_dataframes

    @property
    def supervised_dataframes(self):
        return self.__supervised_dataframes

    @property
    def extract_samples(self):
        return self.__extract_samples

    @extract_samples.setter
    def extract_samples(self, fun):
        self.__extract_samples = fun

    def scale_dataframes(self, threshold, scale_range, scale_features=None):
        if scale_features is None:
            scale_features = ['sgv']
        if self.__data_frames is None:
            raise ValueError("No Dataframes set yet!")

        lower_bound, upper_bound = threshold
        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound

        if self.scalers is None:
            self.scalers = {}
            for column in self.__data_frames[list(self.data_frames.keys())[0]][0].columns:
                if column in scale_features:
                    self.scalers[column] = MinMaxScaler(feature_range=scale_range)
                    self.scalers[column].fit(np.array([upper_bound, lower_bound]).reshape(-1, 1))
        self.__scaled_dataframes = defaultdict(list)
        for key, dfs in self.__data_frames.items():
            for df in dfs:
                self.__scaled_dataframes[key].append(scale_dataframe(df, self.scalers))

        return self.__scaled_dataframes

    def roll_dataframes(self, window_size, skips):
        if self.__scaled_dataframes is None:
            raise ValueError("Dataframes not scaled yet!")

        self.__rolled_dataframes = defaultdict(list)
        for key, dfs in self.__scaled_dataframes.items():
            for df in dfs:
                self.__rolled_dataframes[key].extend(roll_dataframe(df, window_size, skips))

        return self.__rolled_dataframes

    def convert2supervised(self, n_in, n_out, skips, rolled=True):
        self.__supervised_dataframes = defaultdict(list)
        if rolled:
            if self.__rolled_dataframes is None:
                raise ValueError("Dataframes not rolled yet!")

            for key, dfs in self.__rolled_dataframes.items():
                for df in dfs:
                    self.__supervised_dataframes[key].append(convert_to_supervised(df, n_in, n_out, skips))
        else:
            if self.__scaled_dataframes is None:
                raise ValueError("Dataframes not scaled yet!")
            for key, dfs in self.__scaled_dataframes.items():
                for df in dfs:
                    self.__supervised_dataframes[key].append(convert_to_supervised(df, n_in, n_out, skips))

        return self.__supervised_dataframes

    def train_test_split(self, data=None, splitpoint=0.66, shuffle=True, split_type=None):
        data = data if data is not None else self.__supervised_dataframes
        train, test = defaultdict(list), defaultdict(list)
        all_data = list()
        for key, dfs in data.items():
            for df in dfs:
                all_data.append((key, df))
        if split_type == "in_samples":
            sp = int(len(all_data) * splitpoint)
            print(sp, len(all_data))
            train_l, test_l, train_le, test_le = 0, 0, 0, 0

            if shuffle:
                np.random.shuffle(all_data)

            for key, df in all_data[:sp]:
                train_le += len(df)
                train_l += 1
                train[key].append(df)
            for key, df in all_data[sp:]:
                test_le += len(df)
                test_l += 1
                test[key].append(df)
            print(train_l, test_l, train_le, test_le)
        elif split_type == "in_frame":
            for key, df in all_data:
                sp = int(len(df) * splitpoint)
                train[key].append(df.iloc[:sp])
                test[key].append(df.iloc[sp:])
        else:
            sp = int(len(data) * splitpoint)
            ids = list(data.keys())
            if shuffle:
                np.random.shuffle(ids)
            train_ids, test_ids = ids[:sp], ids[sp:]

            for key in train_ids:
                train[key] = data[key]
            for key in test_ids:
                test[key] = data[key]
        return train, test

    def data_to_sample(self, data, n_in, n_out):
        if not callable(self.__extract_samples):
            raise AttributeError("No extract_sample function available. Set one first")

        return self.__extract_samples(data, n_in, n_out)

    def routine(self, lower_bound=20, upper_bound=400, scale_range=(-1, 1), window_size=None, roll_skips=None, skips=6,
                split=0.66, val_split=None, n_in=1, n_out=1):
        self.scale_dataframes((lower_bound, upper_bound), scale_range)
        rolling = window_size is not None and roll_skips is not None
        if rolling:
            self.roll_dataframes(window_size + skips + n_in + n_out - 1, roll_skips)
        self.convert2supervised(n_in, n_out, skips, rolling)
        data = self.__supervised_dataframes
        train, test = self.train_test_split(data, split)
        if val_split is not None:
            train, val = self.train_test_split(train, val_split)
            (train_x, train_y, train_dic), (test_x, test_y, test_dic), (val_x, val_y, val_dic) = \
                self.data_to_sample(train, n_in, n_out), \
                self.data_to_sample(test, n_in, n_out), \
                self.data_to_sample(val, n_in, n_out)
            return (train_x, train_y, train_dic), (test_x, test_y, test_dic), (val_x, val_y, val_dic)

        (train_x, train_y, train_dic), (test_x, test_y, test_dic) = self.data_to_sample(train, n_in, n_out), self.data_to_sample(test, n_in,
                                                                                                            n_out)

        return (train_x, train_y, train_dic), (test_x, test_y, test_dic)

    def plot_dataframes(self, original=False, scaled=False, rolled=False, title="", x_range=4, figsize=(16, 10)):
        xor = 0
        conds = [original, scaled, rolled]
        for cond in conds:
            if cond is True:
                xor += 1

        if not xor == 1:
            raise AttributeError("Only one kind or at least one of Dataframe can be plotted at once")

        dataframes = None
        if original:
            dataframes = self.__data_frames
        elif scaled:
            dataframes = self.__scaled_dataframes
        elif rolled:
            dataframes = self.__rolled_dataframes

        y_lim = (20, 400) if original else (-1.5, 1.5)
        h_span = (70, 200) if original else (-0.73684211, -0.05263158)

        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(hspace=.32, wspace=.15)
        fig.suptitle(title)
        count = 1
        n_ts = len(dataframes)

        for dataframe in dataframes:
            x, y = dataframe.index.to_pydatetime(), dataframe.values

            ax = fig.add_subplot(int((n_ts + (x_range - n_ts % x_range)) / x_range), x_range, count)
            ax.set_facecolor("#DDE4F6")
            ax.grid(color="w", linestyle="solid")
            ax.set_ylim(y_lim[0], y_lim[1])
            ax.set_xlim(x[0], x[-1])
            plt.xticks(rotation=30)
            ax.axhspan(h_span[0], h_span[1], alpha=0.1, color="green")
            ax.plot(x, y)

            count += 1

        plt.legend()
        plt.show()
