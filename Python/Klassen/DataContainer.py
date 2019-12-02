import copy
import random

import dateutil.parser
import dateutil.tz
import helperfunctions
import json
import numpy as np
import pandas as pd
import warnings

from collections import defaultdict
from datetime import timedelta, datetime
from helperclasses import ProgressBar
from matplotlib.dates import date2num
from pykalman import KalmanFilter
from scipy.interpolate import splev
from scipy.interpolate import splrep
from helperclasses import CustomDateParser
from helperfunctions import roll_dataframe


class DataContainer:
    def __init__(self):
        self.__entries = list()
        self.__treatments = {}
        self.__profile = {}
        self.__devicestatus = list()
        self.__dateEntries = []
        self.date_parser = CustomDateParser(['en'])
        self.__imported_entries = 0
        self.__imported_treatments = 0

    def __len__(self):
        return len(self.__entries)

    def importEntries(self, file, debug=False, threshold=(400.0, 20.0), show_warnings=False):
        if not file[-4:] == "json":
            raise Exception('File needs to be in json format')
        with open(file) as jsonfile:
            if len(self.__entries) == 0:
                self.__entries = json.load(jsonfile)
            else:
                self.__entries.extend(json.load(jsonfile))
        self.__cleanUpEntryDates(debug, threshold, show_warnings=show_warnings)

    def importTreatments(self, file, debug=False, show_warnings=False):
        if not file[-4:] == "json":
            raise Exception('File needs to be in json format')
        with open(file) as jsonfile:
            if len(self.__treatments) == 0:
                self.__treatments = json.load(jsonfile)
            else:
                self.__treatments.extend(json.load(jsonfile))

        self.__cleanUpTreatments(show_warnings=show_warnings)

    def importProfile(self, file):
        if not file[-4:] == "json":
            raise Exception('File needs to be in json format')
        with open(file) as jsonfile:
            self.__profile = json.load(jsonfile)

    def importDeviceStatus(self, file, debug=False, show_warnings=False):
        if not file[-4:] == "json":
            raise Exception('File needs to be in json format')
        with open(file) as jsonfile:
            if len(self.__devicestatus) == 0:
                self.__devicestatus = json.load(jsonfile)
            else:
                self.__devicestatus.extend(json.load(jsonfile))

        self.__cleanUpDeviceStatus(show_warnings)

    def __cleanUpEntryDates(self, debug, threshold, duplicate_handling='mean', show_warnings=False):
        iterations = len(self.__entries)
        indices_to_remove = list()
        upper_bound, lower_bound = threshold

        if self.date_parser.timezone is None:
            self.find_timezone(show_warnings=show_warnings)

        pb = ProgressBar("Loading {} Entries".format(
            iterations),
            iterations) if self.__imported_entries == 0 else ProgressBar(
            "Adding {} Entries".format(iterations - self.__imported_entries),
            iterations)

        previous_entry = None
        for i, entry in enumerate(self.__entries):
            pb.update(i + 1)

            if type(entry['date']) == datetime:
                continue

            if not entry['type'] == 'sgv':
                indices_to_remove.append(i)
                continue

            # Convert SGV Value into float
            entry['sgv'] = float(entry['sgv'])

            if entry['sgv'] > upper_bound or entry['sgv'] < lower_bound:
                indices_to_remove.append(i)
                continue

            datetime_string = entry['dateString']

            date = self.date_parser.parse(datetime_string, show_warnings=show_warnings)

            offset = date.utcoffset().total_seconds()

            date = date - timedelta(seconds=offset)
            date = date.replace(tzinfo=dateutil.tz.tzutc())
            date = date + helperfunctions.roundedSeconds(date)
            entry['date'] = date

            if previous_entry is not None:
                previous_date = previous_entry['date']
                if previous_date == date:
                    previous_entry['sgv'] = (previous_entry['sgv'] + entry['sgv']) / 2.0
                    indices_to_remove.append(i)
                    continue

            previous_entry = entry

        pb.done()

        self.remove_by_index(indices_to_remove)

        self.__entries = sorted(self.__entries, key=lambda x: x['date'], reverse=True)

        previous_entry = None
        indices_to_remove = list()
        for i, entry in enumerate(self.__entries):
            if previous_entry is None:
                previous_entry = entry
                continue

            if previous_entry['date'] == entry['date']:
                if not previous_entry['sgv'] == entry['sgv']:
                    if show_warnings:
                        warnings.warn(
                            "Two different SGV Values at {} (DateString: {}) -> {} and {} (DateString: {}) -> {} Using Duplicate Handling {}".format(
                                previous_entry['date'],
                                previous_entry['dateString'],
                                previous_entry['sgv'],
                                entry['date'],
                                entry['dateString'],
                                entry['sgv'],
                                str(duplicate_handling)))
                    if duplicate_handling == 'mean':
                        entry['sgv'] = (entry['sgv'] + previous_entry['sgv']) / 2.0
                    elif duplicate_handling == "first":
                        entry['sgv'] = previous_entry['sgv']
                    elif duplicate_handling == "last":
                        pass
                    else:
                        raise Exception("Unknown Duplicate Handler {}".format(str(duplicate_handling)))

                indices_to_remove.append(i - 1)
                continue

            previous_entry = entry

        self.remove_by_index(indices_to_remove, message="Removing {} duplicate entries. Started off with {} "
                                                        "entries. Remaining: {}".format(len(indices_to_remove),
                                                                                        len(self.__entries),
                                                                                        len(self.__entries) - len(
                                                                                            indices_to_remove)))

        self.__imported_entries = len(self.__entries)

    def __cleanUpDeviceStatus(self, show_warnings=False):
        valid = [status for status in self.__devicestatus if 'date' in status.keys()]
        pb = ProgressBar("Extract valid status from {} Devicestatus".format(
            len(self.__devicestatus) - len(valid)),
            len(self.__devicestatus) - len(valid))
        for i, status in enumerate(self.__devicestatus[len(valid):]):
            pb.update(i + 1)
            if 'openaps' in status.keys() or 'predBGs' in status.keys():
                openaps = status['openaps']
                if 'suggested' in openaps.keys():
                    suggested = openaps['suggested']
                    if 'predBGs' in suggested:
                        valid.append(suggested)
        pb.done()

        pb = ProgressBar("Fix Dates for {} status".format(len(valid)), len(valid))
        for i, status in enumerate(valid):
            pb.update(i + 1)
            if 'date' in status.keys():
                continue
            date_string = status['timestamp']
            date = self.date_parser.parse(date_string, show_warnings=show_warnings)
            offset = date.utcoffset().total_seconds()
            date = date - timedelta(seconds=offset)
            date = date.replace(tzinfo=dateutil.tz.tzutc())
            status['date'] = date
        pb.done()

        self.__devicestatus = valid

    def __cleanUpTreatments(self, debug=False, show_warnings=False):
        if len(self.__entries) == 0:
            warnings.warn("No Entries found")
            return
        if self.date_parser.timezone is None:
            warnings.warn("Need to find Timezone first Parse Entries")
            return

        iterations = len(self.__treatments)
        pb = ProgressBar("Importing {} Treatments".format(
            iterations),
            iterations) if self.__imported_treatments == 0 else ProgressBar(
            "Adding {} Treatments".format(iterations - self.__imported_treatments),
            iterations)

        previous_treatment = None
        indices_to_remove = list()
        for i, treatment in enumerate(self.__treatments):
            pb.update(i + 1)

            datetime_string = treatment['created_at']
            date = self.date_parser.parse(datetime_string, show_warnings=show_warnings)
            offset = date.utcoffset().total_seconds()
            date = date - timedelta(seconds=offset)
            date = date.replace(tzinfo=dateutil.tz.tzutc())
            treatment['date'] = date

        self.__treatments = sorted(self.__treatments, key=lambda x: x['date'], reverse=True)

        for i, treatment in enumerate(self.__treatments):
            try:
                insulin = treatment['insulin']
            except KeyError:
                insulin = 0
                treatment['insulin'] = insulin
            try:
                carbs = treatment['carbs']
            except KeyError:
                carbs = 0
                treatment['carbs'] = carbs
            current_date = treatment['date']

            if previous_treatment is None:
                previous_treatment = treatment
                continue

            previous_date = previous_treatment['date']
            previous_insulin = previous_treatment['insulin']
            previous_carbs = previous_treatment['carbs']
            if previous_date == current_date:
                if not previous_insulin == insulin and previous_insulin is not None and insulin is not None:
                    if show_warnings:
                        warnings.warn(
                            "Same Date but different values for Insulin: {} <-> {}".format(previous_insulin, insulin))
                    treatment['insulin'] = (insulin + previous_insulin) / 2
                else:
                    treatment['insulin'] = previous_insulin if previous_insulin is not None else insulin
                if not previous_carbs == carbs and previous_carbs is not None and carbs is not None:
                    if show_warnings:
                        warnings.warn(
                            "Same Date but different values for Carbs: {} <-> {}".format(previous_carbs, carbs))
                    treatment['carbs'] = (carbs + previous_carbs) / 2
                else:
                    treatment['carbs'] = previous_carbs if previous_carbs is not None else carbs

                indices_to_remove.append(i - 1)
                continue

            previous_treatment = treatment

        pb.done()
        self.remove_by_index(indices_to_remove, data=self.__treatments,
                             message="Removing {} duplicate treatmens. Started off with {} "
                                     "treatments. Remaining: {}".format(len(indices_to_remove),
                                                                        len(self.__treatments),
                                                                        len(self.__treatments) - len(
                                                                            indices_to_remove)))

        self.__imported_treatments = len(self.__treatments)

    def mergeTreatments(self):
        if len(self.__entries) == 0:
            print("No Entries to merge")
            return
        pb = ProgressBar("Merging Treatments into Entries", len(self.__entries))
        if len(self.__treatments) == 0:
            for i, entry in enumerate(self.__entries):
                pb.update(i+1)
                entry['insulin'] = 0
                entry['carbs'] = 0
            pb.done()
            return


        treatments_dates = [t['date'] for t in self.__treatments]
        treatment_indexes = helperfunctions.closest_date(dates=treatments_dates, return_dataframe=True)
        for i, entry in enumerate(self.__entries):
            pb.update(i + 1)

            entry_date = entry['date']
            cd, index = helperfunctions.closest_date(entry_date, treatment_indexes, return_index=True)
            if abs((entry_date - cd).total_seconds()) > 0.5 * 5 * 60:
                entry['insulin'] = 0
                entry['carbs'] = 0
                continue

            treatment = self.__treatments[index]
            insulin = treatment['insulin'] if treatment['insulin'] is not None else 0
            carbs = treatment['carbs'] if treatment['carbs'] is not None else 0
            if 'insulin' not in entry.keys():
                entry['insulin'] = insulin
            elif entry['insulin'] < insulin:
                entry['insulin'] = insulin
            if 'carbs' not in entry.keys():
                entry['carbs'] = carbs
            elif entry['carbs'] < carbs:
                entry['carbs'] = carbs

        pb.done()

    """
    This Method can calculate the true values for the given Prediction Range of the openaps device status
    
    
    def calc_device_ytrue(self, dataframes, show_warnings=False):
        for dataframe in dataframes:
            valid_status = [status for status in self.__devicestatus
                            if dataframe.index[-1] > status['date'] + timedelta(minutes=5) > dataframe.index[0]
                            and 'predBGs' in status.keys()]

            x = np.asarray([date2num(d) for d in dataframe.index])
            y = dataframe.values
            spl = splrep(x, y)

            for status in valid_status:
                predBGs = status['predBGs']
                date = status['date']
                newPred = predBGs.copy()
                for key, predictions in predBGs.items():
                    label = key + "_true"
                    y_true = list()
                    for i in range(len(predictions)):
                        y_true.append(splev(date2num(date + timedelta(minutes=5 * (i + 1))), spl))
                    newPred[label] = y_true
                status['predBGs'] = newPred
    """

    def remove_by_index(self, indices_to_remove, data=None, message=None):
        if data is None:
            data = self.__entries

        if message is None:
            message = "Removing {} invalid entries. Started off with {} entries. Remaining: {}".format(
                len(indices_to_remove),
                len(data),
                len(data) - len(indices_to_remove))

        pb = ProgressBar(message,
                         len(indices_to_remove))
        for i, j in enumerate(sorted(indices_to_remove)):
            pb.update(i)
            del data[j - i]
        pb.done()

    def find_timezone(self, show_warnings=False):
        pb = ProgressBar("Finding Timezone Information", len(self.__entries))

        for i, entry in enumerate(self.__entries):
            pb.update(i)
            if dateutil.parser.parse(entry['dateString']).tzinfo is None:
                continue
            else:
                self.date_parser.parse(entry['dateString'], show_warnings=show_warnings)
                break
        if self.date_parser.timezone is None:
            for e in random.sample(self.__entries, int(len(self.__entries) * 0.05)):
                self.date_parser.parse(e['dateString'], show_warnings=show_warnings)
                if self.date_parser.timezone is not None:
                    break
        pb.done("Found Timezone: {}".format(self.date_parser.timezone is not None))

    def getEntries(self, year=None, month=None, day=None):
        if not year and not month and not day:
            warnings.warn("You did not specify the Date! Returning the whole dataset. If you plot the data it is "
                          "probably a mess!")
            return copy.deepcopy(self.__entries)

        entry_list = []
        entries = copy.deepcopy(self.__entries)
        pb = ProgressBar("Get Entries", len(entries))
        for i, entry in enumerate(entries):
            pb.update(i)
            date = entry['date']

            if year is not None and month is not None and day is not None:
                if date.year == year and date.month == month and date.day == day:
                    entry_list.append(entry)
            elif year is None:
                if month is not None and day is not None:
                    if date.month == month and date.day == day:
                        entry_list.append(entry)
                elif month is None:
                    if date.day == day:
                        entry_list.append(entry)
                elif day is None:
                    if date.month == month:
                        entry_list.append(entry)

            elif month is None:
                if year is not None and day is not None:
                    if date.year == year and date.day == day:
                        entry_list.append(entry)
                elif day is None:
                    if date.year == year:
                        entry_list.append(entry)

            elif day is None:
                if year is not None and month is not None:
                    if date.year == year and date.month == month:
                        entry_list.append(entry)
        pb.done()

        if not entry_list:
            raise Exception("No Entry found. The Date " + str(day) + "." + str(month) + "." + str(
                year) + " is probably not part of the Dataset")
        return entry_list

    def showEntries(self, year=None, month=None, day=None):
        # Get Entries by Date
        day_entries = self.getEntries(year=year, month=month, day=day)
        # Extract Date and SGV from Data
        dates = [entry['date'] for entry in day_entries]
        sgv_values = [entry['sgv'] for entry in day_entries]

        # Plot the Data
        helperfunctions.plotData((dates, sgv_values))

    def showEqEntries(self, year=None, month=None, day=None):
        eq_day_entries = self.getEqEntries(year=year, month=month, day=day)
        helperfunctions.plotData((list(eq_day_entries.keys()), list(eq_day_entries.values())))
        # Plot the Data

    def getEqEntries(self, year=None, month=None, day=None):
        """Create Date-sgv_value pairs where the timesteps are equally spaced
        """
        day_entries = self.getEntries(year=year, month=month, day=day)

        dependant_time_steps = helperfunctions.getDependantDates(day_entries)
        sgv_values = helperfunctions.interpolateTimeSteps(dependant_time_steps)

        return sgv_values

    def getEntryGroups(self, min_length, tolerance):
        """Returns Packs of Entries which are at least minLength long with a tolerant distance of tolerance in seconds
        """
        entry_groups = []
        dropped_groups = 0
        dropped_entries = 0
        found_entries = 0

        entries = copy.deepcopy(self.__entries)

        start_entry = None
        prev_entry = None
        group_of_entries = []
        pb = ProgressBar("Getting EntryGroups", len(entries))
        for i, entry in enumerate(entries):
            pb.update(i)
            if not prev_entry:
                start_entry = entry
                prev_entry = entry
                group_of_entries.append(entry)
                continue

            try:

                time_difference = abs((entry['date'] - prev_entry['date']).total_seconds())
            except Exception as e:
                print(entry, prev_entry)
                print(entry['date'].tzinfo, prev_entry['date'].tzinfo)
                raise e
            if abs(time_difference - 300.0) > tolerance:
                if len(group_of_entries) > min_length:
                    entry_groups.append(group_of_entries)
                    found_entries += len(group_of_entries)
                else:
                    dropped_groups += 1
                    dropped_entries += len(group_of_entries)

                start_entry = entry
                group_of_entries = [start_entry]

            else:
                group_of_entries.append(entry)

            prev_entry = entry
        pb.done()

        print(
            "Dropped a total of %d entry Groups containing %d entries. Found a total of %d Groups containing %d entries" % (
                dropped_groups, dropped_entries, len(entry_groups), found_entries))

        return entry_groups

    def entryGroup2DataFrame(self, entry_group, features=None, inteprolate_features=None, old=False):

        if inteprolate_features is None:
            inteprolate_features = ['sgv']
        if features is None:
            features = ['sgv']

        dates = [entry['date'] for entry in entry_group]
        dates = list(reversed(dates))
        df = pd.DataFrame(dates, columns=['date'])
        for feature in features:
            values = [entry[feature] for entry in entry_group]
            values = list(reversed(values))
            df[feature] = values
        df['datetime'] = pd.to_datetime(df['date'])
        df = df.set_index('datetime')
        df.drop(['date'], axis=1, inplace=True)

        if old:
            return df

        date_range = pd.date_range(start=df.index[0], end=df.index[-1], freq='5T')

        x = np.asarray([date2num(d) for d in df.index])
        x2 = np.asarray([date2num(d) for d in date_range])
        new_dates = [d for d in date_range]
        aligned_data_frame = pd.DataFrame(new_dates, columns=['date'])

        aligned_data_frame['datetime'] = pd.to_datetime(aligned_data_frame['date'])
        aligned_data_frame = aligned_data_frame.set_index('datetime')
        aligned_data_frame.drop(['date'], axis=1, inplace=True)

        for feature in inteprolate_features:
            y = df[feature]
            spl = splrep(x, y)
            y2 = splev(x2, spl)
            aligned_data_frame[feature] = y2

        for feature in features:
            if feature in inteprolate_features:
                continue
            indices = df[feature].index
            tmp_df = pd.DataFrame([], index=aligned_data_frame.index)
            tmp_df[feature] = 0
            for date in indices:
                closest_date = tmp_df.index[tmp_df.index.get_loc(date, method='nearest')]
                tmp_df.loc[closest_date] = df[feature][date]
            aligned_data_frame[feature] = tmp_df[feature].values

        return aligned_data_frame
