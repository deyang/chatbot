__author__ = 'Deyang'


class QueryState(object):

    def __init__(self, raw_query):
        self.raw_query = raw_query


class QueryEngine(object):

    def __init__(self):
        pass

    def execute_query(self, raw_query):
        return None