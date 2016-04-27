import os
from gensim import similarities
from ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model import p_stemmer
import json

from ir_query_engine import engine_logger


__author__ = 'Deyang'

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
SET_FILE_PATH = os.path.join("ir_query_engine", "saved_models", 'topic_words.set')
SIMMX_FILE_PATH = os.path.join("ir_query_engine", "saved_models", 'topic_words.simmx')


def get_set_path():
    return SET_FILE_PATH


class TopicWordLookupModelStruct(object):

    def __init__(self, tfidf_model_struct, topic_word_set, simmx):
        # Topic model mush have a tfidf model struct
        self.tfidf_model_struct = tfidf_model_struct
        self.topic_word_set = topic_word_set
        self.simmx = simmx

    def get_similarities(self, query_doc, compare_docs):
        query_vec = self.get_topic_word_vec(query_doc)
        # print query_vec
        compare_vecs = [self.get_topic_word_vec(doc) for doc in compare_docs]
        sim_mx = similarities.SparseMatrixSimilarity(compare_vecs, num_features=len(self.tfidf_model_struct.dictionary))
        sims = sim_mx[query_vec]
        return list(enumerate(sims))

    @classmethod
    def write_set(cls, topic_word_set):
        path = get_set_path()
        with open(path, 'w') as f:
            f.write(json.dumps(list(topic_word_set)))

    @classmethod
    def read_set(cls):
        path = get_set_path()
        with open(path, 'r') as f:
            c = f.read()
            return set(json.loads(c))

    def get_topic_word_vec(self, raw_doc):
        raw_doc = raw_doc.lower()
        tf_vec = self.tfidf_model_struct.get_tf_vec(raw_doc)
        topic_word_vec = []
        for pair in tf_vec:
            tokenid = pair[0]
            if tokenid in self.topic_word_set:
                topic_word_vec.append((tokenid, pair[1]))

        return topic_word_vec

    def query(self, raw_doc, limit=3):
        vec = self.get_topic_word_vec(raw_doc)

        sims = self.simmx[vec]
        results = list(enumerate(sims))
        results.sort(key=lambda t: t[1], reverse=True)
        return results[:limit]


    @staticmethod
    def normalize_word(raw_word):
        return p_stemmer.stem(raw_word.lower())

    @classmethod
    def get_model(cls, tfidf_model_struct, data_store=None, regen=False):
        md_file_path = get_set_path()
        if not os.path.isfile(md_file_path) or regen:
            engine_logger.info("Generating topic word lookup model")
            topic_word_set = set([])
            for pair in data_store.topic_word_docs:
                topic_words = pair[1]

                for topic_word in topic_words:
                    normed_w = cls.normalize_word(topic_word)
                    tokenid = tfidf_model_struct.dictionary.token2id.get(normed_w, None)
                    if tokenid is not None:
                        topic_word_set.add(tokenid)

            # saving
            cls.write_set(topic_word_set)
            instance = TopicWordLookupModelStruct(tfidf_model_struct,
                                                  topic_word_set,
                                                  None)

            dictionary, corpus = \
                tfidf_model_struct.docs_to_corpus_tf_idf(data_store.doc_set)

            # extract answer corpus
            question_corpus = [data_store.doc_set[aid] for aid in data_store.question_set]
            question_vecs = [instance.get_topic_word_vec(doc) for doc in question_corpus]
            simmx = \
                similarities.SparseMatrixSimilarity(question_vecs,
                                                    num_features=len(tfidf_model_struct.dictionary))

            instance.simmx = simmx
            simmx.save(SIMMX_FILE_PATH)
            return instance
        else:
            engine_logger.info("Loading existing topic word set")
            topic_word_set = cls.read_set()
            simmx = similarities.SparseMatrixSimilarity.load(SIMMX_FILE_PATH)

            return TopicWordLookupModelStruct(tfidf_model_struct, topic_word_set, simmx)

