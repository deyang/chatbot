import os
import sys
import json
from ir_query_engine import engine_logger
import re
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora

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
        self.qa_context = dict()
        # set of question doc ids
        self.question_set = list()
        # set of question doc ids
        self.answer_set = list()
        # question has only one answer
        self.qid_to_qa_pair = dict()
        # one answer can corresponds to multiple questions
        self.aid_to_qa_pairs = dict()
        # data to train svm rank
        self.rank_data = dict()

        self.topic_word_docs = []

        for segment in raw_json_object:
            question = segment['_question']
            qid, added = self._add_doc(question)
            if added:
                # new question, add to question set
                self.question_set.append(qid)
                # context
                if 'context' in segment:
                    self.qa_context[qid] = segment['context']

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
            if '_question_topic_words' in segment:
                self.topic_word_docs.append((question, segment['_question_topic_words']))
            if 'answer_topic_words' in segment:
                self.topic_word_docs.append((answer, segment['answer_topic_words']))

        # loop for the ranked answer, add them at last, since not all of them correspond to an question
        for segment in raw_json_object:
            if 'qa_pairs_with_matching_score' in segment and \
                            len(segment['qa_pairs_with_matching_score']) > 0:
                self.rank_data[segment['_question']] = []
                for pair_dict in segment['qa_pairs_with_matching_score']:
                    self._add_doc(pair_dict['_question'])
                    self._add_doc(pair_dict['answer'])
                    self.rank_data[segment['_question']].append((
                        (
                            pair_dict['_question'],
                            pair_dict['answer']
                        ),
                        pair_dict['score'])
                    )
        engine_logger.info("# docs loaded: %s" % len(self.doc_set))
        engine_logger.info("# questions loaded: %s" % len(self.question_set))
        engine_logger.info("# answers loaded: %s" % len(self.answer_set))
        engine_logger.info("# topic word labeled docs: %s" % len(self.topic_word_docs))

    def get_all_questions(self):
        return [self.doc_set[qid] for qid in self.question_set]

    def get_all_answers(self):
        return [self.doc_set[aid] for aid in self.answer_set]

    def get_doc_id_from_answer_pos(self, pos):
        return self.answer_set[pos]

    def get_doc_id_from_question_pos(self, pos):
        return self.question_set[pos]

    def translate_answer_query_results(self, results):
        return [(self.get_doc_id_from_answer_pos(idx), sim) for idx, sim in results]

    def translate_question_query_results(self, results):
        return [(self.get_doc_id_from_question_pos(idx), sim) for idx, sim in results]

    def get_docs_by_pair(self, pair):
        return map(self.doc_set.__getitem__, pair)

    def __repr__(self):
        return "doc sets:\n%s\ntopics words:\n%s" % \
               (json.dumps(self.doc_set, indent=4), json.dumps(self.topic_word_docs, indent=4))


def load_raw_data(file_name):
    path = os.path.join(DATA_PATH, file_name)

    if not os.path.isfile(path):
        engine_logger.error("cannot find the file")
        sys.exit(1)
    with open(path, 'r') as f:
        contents = f.read()
    try:
        json_obj = json.loads(contents)
        return json_obj
    except Exception as e:
        engine_logger.error(e)
        return None

    data_store = DataStore(json_obj)
    return data_store


def load_data_store(file_name):
    raw_data = load_raw_data(file_name)
    if raw_data:
        return DataStore(raw_data)


def sample_by_index_generator(indices, raw_list):
    for index in indices:
        yield raw_list[index]

tokenizer = RegexpTokenizer(r'\w+')
p_stemmer = PorterStemmer()

# create English stop words list
en_stop = get_stop_words('en')
customized_stop_words = [
    'show', 'want', 'know', 'can', 'find', 'tell', 'need', 'information'
]
stop_words = set(en_stop + customized_stop_words)


def pre_process_doc(raw_doc, rm_stop_words=False):
    # clean and tokenize document string
    cleaned_doc = re.sub(r'https?:\/\/.*\s?$', 'http', raw_doc.lower())
    tokens = tokenizer.tokenize(cleaned_doc)

    if rm_stop_words:
        # remove stop words from tokens
        tokens = [t for t in tokens if t not in en_stop]

    # stem tokens
    stemmed_tokens = [p_stemmer.stem(t) for t in tokens]
    return stemmed_tokens


def docs_to_corpus(doc_set, rm_stop_words=False):
    # list for tokenized documents in loop
    texts = []

    # loop through document list
    for i in doc_set:
        # add tokens to list
        texts.append(pre_process_doc(i, rm_stop_words=rm_stop_words))

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)
    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]
    return dictionary, corpus
