from __future__ import unicode_literals
import word2vec
import os
from ir_query_engine.retrieve_match_models.tf_idf_feature.transform import TfIdfModelStruct
from scipy.spatial.distance import cosine

# Use this on production ec2 instance
filename = 'GoogleNews-vectors-negative300.bin'
# Use this on mac dev env
filename = 'vectors.bin'
dirpath = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(dirpath, filename)


class Word2VecModel(object):
    def __init__(self):
        self.model = word2vec.load(filepath, encoding='ISO-8859-1')

    def get_sent_vec(self, raw_doc):
        tokenized_doc = TfIdfModelStruct.pre_process_doc_tf_idf(raw_doc)
        vec_sum = None
        for token in tokenized_doc:
            try:
                vec = self.model[token]
                if vec_sum is None:
                    vec_sum = vec
                else:
                    vec_sum += vec
            except:
                pass
        return vec_sum

    def get_similarities(self, query_doc, compare_docs):
        query_vec = self.get_sent_vec(query_doc)
        compare_vecs = [self.get_sent_vec(doc) for doc in compare_docs]

        sims = []
        for vec in compare_vecs:
            sims.append(1 - cosine(query_vec, vec))
        results = list(enumerate(sims))
        results.sort(key=lambda t: t[1], reverse=True)
        return results
