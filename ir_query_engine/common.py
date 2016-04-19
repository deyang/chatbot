import os
import sys
import json
from ir_query_engine import engine_logger

__author__ = 'Deyang'

dirpath = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(dirpath, '..', 'data')


class DataStore(object):

    def __init__(self, raw_json_object):
        self.doc_set = list()
        for segment in raw_json_object:
            self.doc_set.append(segment['question'])
            self.doc_set.append(segment['answer'])
            if len(segment['ranked_answers']) > 0:
                for ranked_answer in segment['ranked_answers'][1:]:
                    # SKIP the first, which is the same with the best answer
                    self.doc_set.append(ranked_answer['answer'])

    def __repr__(self):
        return json.dumps(self.doc_set, indent=4)


def load_data(file_name):
    path = os.path.join(DATA_PATH, file_name)

    if not os.path.isfile(path):
        engine_logger.error("cannot find the file")
        sys.exit(1)
    with open(path, 'r') as f:
        contents = f.read()
    try:
        json_obj = json.loads(contents)
    except Exception as e:
        engine_logger.error(e)
        return None

    data_store = DataStore(json_obj)
    return data_store
