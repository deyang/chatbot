from __future__ import unicode_literals
import word2vec
import os
from ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model import TfIdfModelStruct
from scipy.spatial.distance import cosine
import numpy
from ir_query_engine import engine_logger

# Use this on production ec2 instance
filename = 'GoogleNews-vectors-negative300.bin'
# Use this on mac dev env
filename = 'vectors.bin'
dirpath = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(dirpath, filename)


class Word2VecModel(object):
    def __init__(self):
        # engine_logger.info("Loading word2vec: %s" % filepath)
        self.model = word2vec.load(filepath, encoding='ISO-8859-1')

    def get_sent_vec(self, raw_doc):
        tokenized_doc = TfIdfModelStruct.pre_process_doc_tf_idf(raw_doc)
        vec_sum = None
        for token in tokenized_doc:
            try:
                vec = self.model[token]
                if vec_sum is None:
                    vec_sum = numpy.array(vec, dtype='float64')
                else:
                    vec_sum += numpy.array(vec, dtype='float64')
            except:
                pass
        return vec_sum

    def get_similarities(self, query_doc, compare_docs):
        query_vec = self.get_sent_vec(query_doc)
        compare_vecs = [self.get_sent_vec(doc) for doc in compare_docs]

        sims = []
        for vec in compare_vecs:
            if vec is None or query_vec is None:
                sims.append(0.0)
            else:
                sims.append(1 - cosine(query_vec, vec))
        return list(enumerate(sims))
