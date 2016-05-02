import os
from gensim import similarities
from ir_query_engine.common import p_stemmer
from ir_query_engine.common import pre_process_doc
from gensim import corpora
from ir_query_engine import engine_logger


__author__ = 'Deyang'

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
SIMMX_FILE_PATH = os.path.join(DIR_PATH, "..", "saved_models", 'topic_words.simmx')
DICT_FILE_PATH = os.path.join(DIR_PATH, "..", "saved_models", 'topic_words.dict')


def get_simmx_path():
    return SIMMX_FILE_PATH


def get_dict_path():
    return DICT_FILE_PATH


class TopicWordLookupModelStruct(object):

    def __init__(self, dictionary, simmx):
        self.dictionary = dictionary
        self.simmx = simmx

    def get_similarities(self, query_doc, compare_docs):
        query_vec = self.get_topic_word_vec(query_doc)
        compare_vecs = [self.get_topic_word_vec(doc) for doc in compare_docs]
        sim_mx = similarities.SparseMatrixSimilarity(compare_vecs, num_features=len(self.dictionary))
        sims = sim_mx[query_vec]
        return list(enumerate(sims))

    def get_topic_word_vec(self, raw_doc):
        pre_processed_doc = pre_process_doc(raw_doc)
        return self.dictionary.doc2bow(pre_processed_doc)

    def query(self, raw_doc=None, limit=10):
        vec = self.get_topic_word_vec(raw_doc)

        sims = self.simmx[vec]
        results = list(enumerate(sims))
        results.sort(key=lambda t: t[1], reverse=True)
        return results[:limit]

    @staticmethod
    def normalize_word(raw_word):
        return p_stemmer.stem(raw_word.lower())

    @classmethod
    def get_model(cls, data_store=None, regen=False, save=True):
        dict_file_path = get_dict_path()
        simmx_file_path = get_simmx_path()
        if not os.path.isfile(dict_file_path) or not os.path.isfile(simmx_file_path) or \
                regen:
            engine_logger.info("Generating topic word lookup model")

            # construct on a refined vocabulary of only topic words
            topic_words_across_all_docs = []
            for pair in data_store.topic_word_docs:
                topic_words_per_doc = pair[1]
                normed_topic_words_per_doc = [cls.normalize_word(word) for word in topic_words_per_doc]
                topic_words_across_all_docs.append(normed_topic_words_per_doc)

            dictionary = corpora.Dictionary(topic_words_across_all_docs)
            instance = TopicWordLookupModelStruct(dictionary,
                                                  None)

            topic_word_vecs = [instance.get_topic_word_vec(doc) for doc in data_store.doc_set]
            simmx = \
                similarities.SparseMatrixSimilarity(topic_word_vecs,
                                                    num_features=len(dictionary))

            instance.simmx = simmx
            if save:
                # saving
                simmx.save(simmx_file_path)
                dictionary.save_as_text(dict_file_path)

            return instance
        else:
            engine_logger.info("Loading existing topic word lookup model")
            dictionary = corpora.Dictionary.load_from_text(dict_file_path)
            simmx = similarities.SparseMatrixSimilarity.load(simmx_file_path)

            return TopicWordLookupModelStruct(dictionary, simmx)
