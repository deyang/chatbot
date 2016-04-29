from gensim import corpora, models, similarities
import os
from ir_query_engine import engine_logger
from utils.util import StopWatch
from ir_query_engine.common import pre_process_doc, docs_to_corpus

__author__ = 'Deyang'


DIR_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE_PATH = os.path.join(DIR_PATH, "..", "..", "saved_models", 'lda.md')
DICT_FILE_PATH = os.path.join(DIR_PATH, "..", "..", "saved_models", 'lda.dict')
SIMMX_FILE_PATH = os.path.join(DIR_PATH, "..", "..", "saved_models", 'lda.simmx')
NUM_TOPIC_FILE_PATH = os.path.join(DIR_PATH, "..", "..", "saved_models", 'lda_num_topics.txt')


def get_md_path():
    return MODEL_FILE_PATH


def get_dict_path():
    return DICT_FILE_PATH


def get_simmx_path():
    return SIMMX_FILE_PATH


def get_num_topic_path():
    return NUM_TOPIC_FILE_PATH


def write_num_topics(filepath, num_topics):
    with open(filepath, 'w') as f:
        f.write(str(num_topics))


def read_num_topics(filepath):
    with open(filepath, 'r') as f:
        c = f.read()
        return int(c)


class LdaModelStruct(object):

    def __init__(self, model=None, dictionary=None, sim_matrix=None, num_topics=None, rm_stop_words=True):
        self.model = model
        self.dictionary = dictionary
        self.sim_matrix = sim_matrix
        self.num_topics = num_topics
        self.rm_stop_words = rm_stop_words

    def get_topic_predict(self, raw_doc):
        return self.model[self.dictionary.doc2bow(pre_process_doc(raw_doc, rm_stop_words=self.rm_stop_words))]

    def get_similarities(self, query_doc, compare_docs):
        # sw = StopWatch()
        query_topic_predict = self.get_topic_predict(query_doc)
        compare_topic_predicts = [self.get_topic_predict(doc) for doc in compare_docs]
        # print "after predict lda: %s" % sw.lap()
        sim_mx = similarities.MatrixSimilarity(compare_topic_predicts, num_features=self.num_topics)
        sims = sim_mx[query_topic_predict]
        # print "total lda sims: %s" % sw.lap()
        return list(enumerate(sims))

    def query(self, topic_predict=None, raw_doc=None, limit=10):
        # Give up the totally not matched results
        if raw_doc:
            topic_predict = self.get_topic_predict(raw_doc)

        sims = self.sim_matrix[topic_predict]
        results = list(enumerate(sims))
        results.sort(key=lambda t: t[1], reverse=True)
        return results[:limit]
        # truncated_results = []
        # for docid, sim in results[:limit]:
        #     if sim > 0.1:
        #         truncated_results.append((docid, sim))
        # return truncated_results

    @classmethod
    def get_model(cls, data_store=None, regen=False, num_topics=None):
        md_file_path = get_md_path()
        dict_file_path = get_dict_path()
        simmx_file_path = get_simmx_path()
        num_topics_file_path = get_num_topic_path()

        if not os.path.isfile(md_file_path) or not \
                os.path.isfile(dict_file_path) or not \
                os.path.isfile(simmx_file_path) or not \
                os.path.isfile(num_topics_file_path) or regen:
            engine_logger.info("Generating LDA models.")

            dictionary, corpus = docs_to_corpus(data_store.doc_set, rm_stop_words=True)
            # generate LDA model
            # LDA model is trained on all the docs
            model = models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)
            sim_matrix = similarities.MatrixSimilarity(model[corpus])

            # saving
            dictionary.save_as_text(dict_file_path)
            model.save(md_file_path)
            sim_matrix.save(simmx_file_path)
            write_num_topics(num_topics_file_path, num_topics)
        else:
            engine_logger.info("Loading existing LDA models.")
            dictionary = corpora.Dictionary.load_from_text(dict_file_path)
            model = models.TfidfModel.load(md_file_path)
            sim_matrix = similarities.SparseMatrixSimilarity.load(simmx_file_path)
            num_topics = read_num_topics(num_topics_file_path)

        return LdaModelStruct(model=model, dictionary=dictionary, sim_matrix=sim_matrix, num_topics=num_topics)


