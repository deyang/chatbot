import os
import sys
import json
from ir_query_engine import engine_logger

__author__ = 'Deyang'

dirpath = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(dirpath, '..', 'data')


class DataStore(object):

    def _add_doc(self, doc):
        """
        If the document is already added, return the corresponding doc id.
         Otherwise, genetate doc id, add doc to doc set and doc to id dict

        :return:
        """
        if doc not in self.doc_to_id:
            # generate doc id
            doc_id = len(self.doc_set)
            # add doc to id mapping
            self.doc_to_id[doc] = doc_id
            # add to doc set
            self.doc_set.append(doc)
            return doc_id, True
        else:
            doc_id = self.doc_to_id[doc]
            return doc_id, False

    def __init__(self, raw_json_object):
        engine_logger.info("Initializing data store.")
        # doc set stores all the documents
        self.doc_set = list()
        self.doc_to_id = dict()
        # set of question doc ids
        self.question_set = list()
        # set of question doc ids
        self.answer_set = list()
        # question has only one answer
        self.qid_to_qa_pair = dict()
        # one answer can corresponds to multiple questions
        self.aid_to_qa_pairs = dict()

        self.topic_word_docs = []

        for segment in raw_json_object:
            question = segment['question']
            qid, added = self._add_doc(question)
            if added:
                # new question, add to question set
                self.question_set.append(qid)

            answer = segment['answer']
            aid, added = self._add_doc(answer)
            if added:
                # new answer, add to answer set
                self.answer_set.append(aid)

            self.qid_to_qa_pair[qid] = (qid, aid)
            if aid not in self.aid_to_qa_pairs:
                self.aid_to_qa_pairs[aid] = []
            self.aid_to_qa_pairs[aid].append((qid, aid))

            # store data for topic words
            self.topic_word_docs.append((question, segment['question_topic_words']))
            self.topic_word_docs.append((answer, segment['answer_topic_words']))

        # loop for the ranked answer, add them at last, since not all of them correspond to an question
        for segment in raw_json_object:
            if len(segment['ranked_answers']) > 0:
                for ranked_answer in segment['ranked_answers'][1:]:
                    # SKIP the first, which is the same with the best answer
                    self._add_doc(ranked_answer['answer'])

    def __repr__(self):
        return "doc sets:\n%s\ntopics words:\n%s" % \
               (json.dumps(self.doc_set, indent=4), json.dumps(self.topic_word_docs, indent=4))


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
