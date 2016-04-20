from gensim import corpora, models, similarities
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import os
from ir_query_engine import engine_logger

__author__ = 'Deyang'

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE_PATH = os.path.join(DIR_PATH, 'tfidf.md')
DICT_FILE_PATH = os.path.join(DIR_PATH, 'tfidf.dict')
SIMMX_FILE_PATH = os.path.join(DIR_PATH, 'tfidf.simmx')


def get_md_path():
    return MODEL_FILE_PATH


def get_dict_path():
    return DICT_FILE_PATH


def get_simmx_path():
    return SIMMX_FILE_PATH

tokenizer = RegexpTokenizer(r'\w+')
p_stemmer = PorterStemmer()


class TfIdfModelStruct(object):

    def __init__(self, model=None, dictionary=None, sim_matrix=None):
        self.model = model
        self.dictionary = dictionary
        self.sim_matrix = sim_matrix

    @staticmethod
    def pre_process_doc_tf_idf(raw_doc):
        """
        Pre-processing fortf-idf does not
        1) rm stop words
        2) rm low-fre words

        :param raw_doc:
        :return:
        """
        # clean and tokenize document string
        tokens = tokenizer.tokenize(raw_doc.lower())
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(t) for t in tokens]
        return stemmed_tokens

    @classmethod
    def docs_to_corpus_tf_idf(cls, doc_set):
        # list for tokenized documents in loop
        texts = []

        # loop through document list
        for i in doc_set:
            # add tokens to list
            texts.append(cls.pre_process_doc_tf_idf(i))

        # turn our tokenized documents into a id <-> term dictionary
        dictionary = corpora.Dictionary(texts)
        # convert tokenized documents into a document-term matrix
        corpus = [dictionary.doc2bow(text) for text in texts]
        return dictionary, corpus

    def get_tfidf_vec(self, raw_doc):
        return self.model[self.dictionary.doc2bow(self.pre_process_doc_tf_idf(raw_doc))]

    def get_tf_vec(self, raw_doc):
        """
        TF vector is bag of words vector
        :param raw_doc:
        :return:
        """
        return self.dictionary.doc2bow(self.pre_process_doc_tf_idf(raw_doc))

    def get_idf_vec(self, raw_doc=None, tf_vec=None):
        if raw_doc:
            tf_vec = self.get_tf_vec(raw_doc)
        return [(termid, self.model.idfs.get(termid))
                for termid, tf in tf_vec if self.model.idfs.get(termid, 0.0) != 0.0]

    def get_tf_and_idf(self, raw_doc, term):
        tf_vec = self.get_tf_vec(raw_doc)
        idf_vec = self.get_idf_vec(tf_vec=tf_vec)
        termid = self.dictionary.token2id.get(term, None)
        # defaults to zero
        tf = 0
        idf = 10.0

        if termid is not None:
            for tid, t_tf in tf_vec:
                if tid == termid:
                    tf = t_tf
            for tid, t_idf in idf_vec:
                if tid == termid:
                    idf = t_idf
        return tf, idf

    def query(self, tfidf_vec=None, raw_doc=None, limit=10):
        if raw_doc:
            tfidf_vec = self.get_tfidf_vec(raw_doc)

        sims = self.sim_matrix[tfidf_vec]
        results = list(enumerate(sims))
        results.sort(key=lambda t: t[1], reverse=True)
        return results[:limit]

    @classmethod
    def get_model(cls, data_store=None, regen=False, save=True):
        md_file_path = get_md_path()
        dict_file_path = get_dict_path()
        simmx_file_path = get_simmx_path()
        if not os.path.isfile(md_file_path) or not \
                os.path.isfile(dict_file_path) or not \
                os.path.isfile(simmx_file_path) or regen:
            engine_logger.info("Generating TF_IDF models.")

            dictionary, corpus = cls.docs_to_corpus_tf_idf(data_store.doc_set)
            model = models.TfidfModel(corpus)
            sim_matrix = similarities.SparseMatrixSimilarity(model[corpus], num_features=len(dictionary))

            if save:
                # saving
                dictionary.save_as_text(dict_file_path)
                model.save(md_file_path)
                sim_matrix.save(simmx_file_path)
        else:
            engine_logger.info("Loading existing TF_IDF models.")

            dictionary = corpora.Dictionary.load_from_text(dict_file_path)
            model = models.TfidfModel.load(md_file_path)
            sim_matrix = similarities.SparseMatrixSimilarity.load(simmx_file_path)

        return TfIdfModelStruct(model=model, dictionary=dictionary, sim_matrix=sim_matrix)
