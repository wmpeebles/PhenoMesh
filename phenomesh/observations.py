import datetime


class Observation:
    def __init__(self, data=None, year=0, month=0, day=0, hour=0, minute=0, second=0, microsecond=0):
        """
        Observations are defined by data collected at a specific time
        :param year:
        :param month:
        :param day:
        :param hour:
        :param minute:
        :param second:
        :param microsecond:
        :param data: Can be any type of data
        """
        self.datetime = self.set_datetime(year, month, day, hour, minute, second, microsecond)
        self.data = self.set_data(data)

    def set_datetime(self, year, month, day, hour, minute, second, microsecond):
        self.datetime = datetime.datetime(year, month, day, hour, minute, second, microsecond)
        return self.datetime

    def set_data(self, data):
        self.data = data
        return self.data

    def view(self):
        """
        Method for viewing observations
        :return:
        """
        if hasattr(self.data, 'view'):
            self.data.view()
        else:
            print(self.data)
