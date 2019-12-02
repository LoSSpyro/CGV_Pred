import sys, time
import datetime
import warnings

from collections import MutableSequence

from dateparser import timezones


class NearestNeighbourDateList(MutableSequence):
    def __init__(self, *args):
        self.list = list()
        self.extend(list(args))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i):
        return self.list[i]

    def __delitem__(self, i):
        del self.list[i]

    def __setitem__(self, i, v):
        self.list.insert(i, v)

    def insert(self, i, v):
        self.list.insert(i, v)

    def append(self, value, anchor=None):
        if len(self.list) < 2:
            self.list.append(value)
        elif not anchor:
            raise Exception("Need Anchor value to make Comparison.")
        else:
            diff = (value['date'] - anchor).total_seconds()
            if diff < 0:
                for item in self.list:
                    if (item['date'] - anchor).total_seconds() < diff:
                        self.list.remove(item)
                        self.list.append(value)
                        self.list.sort()
                        diff = (item['date'] - anchor).total_seconds()
            elif diff > 0:
                for item in self.list:
                    if (item['date'] - anchor).total_seconds() > diff:
                        self.list.remove(item)
                        self.list.append(value)
                        self.list.sort()
                        diff = (item['date'] - anchor).total_seconds()
            else:
                self.list = [value, value]

    def __str__(self):
        return str(self.list)


class ProgressBar:
    def __init__(self, name, iterations, bar_length=50):
        self.name = name
        self.barLength = bar_length
        self.status = ""
        self.progress = 0
        self.started = False
        self.iterations = iterations if iterations > 0 else 1
        self.last_progress = 0.0
        self.prev_sec = 0

        self.startTime = None

    def start(self):
        sys.stdout.write("\nStart process: %s\n" % self.name)
        self.startTime = datetime.datetime.now()
        self.started = True

    def update(self, progress, message=""):
        if not self.started:
            self.start()
        self.progress = float(progress) / self.iterations
        sec = 0
        if self.progress - self.last_progress > 0.001 or self.progress >= 1.0 or self.progress < 0.0:
            if progress > 0:
                now = datetime.datetime.now()
                left = (self.iterations - progress) * (now - self.startTime) / progress
                sec = int(left.total_seconds())
            if self.progress < 0.0:
                self.progress = 0.0
                self.status = "Halt...\r"
            if self.progress >= 1.0:
                self.progress = 1.0
                self.status = "Done...\r"

            self.last_progress = self.progress
            self.prev_sec = sec
            self.display(sec, message=message)

    def done(self, message="\n"):
        if self.startTime is not None:
            finish_time = datetime.datetime.now() - self.startTime
            finish_time = str(datetime.timedelta(seconds=int(finish_time.total_seconds())))
            message = "It took " + finish_time + " " + message
        self.update(self.iterations + 1, message)

    def display(self, remaining, message=""):
        block = int(round(self.barLength * self.progress))
        time_left = "" if remaining is 0 else str(datetime.timedelta(seconds=remaining))
        text = "\rProgress: [{0}] {1:.2f}% {2} Remaining Time: {3} {4}".format(
            "#" * block + "-" * (self.barLength - block), self.progress * 100, self.status, time_left, message)
        sys.stdout.write(text)
        sys.stdout.flush()


import dateparser
import datetime as dt
import dateutil.parser
import dateutil.tz
from dateparser import timezone_parser


class CustomDateParser:

    def __init__(self, languages):
        self.parser = dateparser.DateDataParser(languages=languages)
        self.timezone = None
        self.changed = False

    def parse(self, date_string, show_warnings=False):
        if "GMT+" in date_string or "GMT-" in date_string:
            ts = date_string.split()
            tz = ts.pop(4)
            tz_offset = int(tz[-6] + str(int(tz[-5:-3]) * 60 + int(tz[-2:])))
            date = dt.datetime.strptime(' '.join(ts), '%a %b %d %H:%M:%S %Y') - dt.timedelta(
                minutes=tz_offset)
            date = date.replace(tzinfo=dt.timezone.utc)
        else:
            try:
                date = dateutil.parser.parse(date_string)
            except Exception as e:
                try:
                    ts = date_string.split()
                    tz = ts.pop(4)
                    tz_offset = int(tz[-6] + str(int(tz[-5:-3]) * 60 + int(tz[-2:])))
                    date = dt.datetime.strptime(' '.join(ts), '%a %b %d %H:%M:%S %Y') - dt.timedelta(
                        minutes=tz_offset)
                    date = date.replace(tzinfo=dateutil.tz.tzutc())

                except Exception as e2:
                    try:
                        date = dt.datetime.strptime(date_string[:-3], '%m/%d/%Y %H:%M:%S')
                    except Exception as e3:
                        try:
                            ts = date_string.split()
                            ts = ts[:-1]
                            date = dateutil.parser.parse(ts)
                        except Exception as e4:
                            print("Failed to interpret date_string {}".format(date_string))
                            print(e4)
                            raise e4

        if date.tzinfo is None:
            _, tz = timezone_parser.pop_tz_offset_from_string(date_string)

            if tz is not None:
                date = date.replace(tzinfo=tz)

        if date.tzinfo is None:
            if self.changed and show_warnings:
                warnings.warn(
                    "Used Saved timezone Information for {} (DateString: {}) despite it changed over time 3".format(
                        str(date), date_string))
            date = date.replace(tzinfo=self.timezone)
        self.update_timezone(date, date.tzinfo, date_string=date_string)
        return date

    def update_timezone(self, date, tz, show_warnings=False, date_string=None):
        if self.timezone is None:
            self.timezone = tz
        elif not self.timezone.utcoffset(date).total_seconds() == tz.utcoffset(date).total_seconds():
            if show_warnings:
                warnings.warn("Changing Timezone from {} to {} for DateString {}".format(self.timezone, tz, date_string))
            self.changed = True
            self.timezone = tz
