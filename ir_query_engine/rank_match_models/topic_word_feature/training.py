from gensim import corpora, models, similarities
from ir_query_engine.retrieve_match_models.tf_idf_feature.transform import TfIdfModelStruct, p_stemmer
import os
from sklearn.externals import joblib
from ir_query_engine import engine_logger
from sklearn.linear_model import LogisticRegression
import nltk
from ir_query_engine.common import DataStore

__author__ = 'Deyang'

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE_PATH = os.path.join(DIR_PATH, 'lr.pkl')


def get_md_path():
    return MODEL_FILE_PATH


0
# create sample documents
doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
doc_e = "Health professionals say that brocolli is good for your health."

# compile sample documents into a list
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]

#
# dictionary, corpus = docs_to_corpus_tf_idf(doc_set)
# tfidf = models.TfidfModel(corpus)

# query part
doc_f = "brocolli is good for healthy."


class TopicWordModelStruct(object):

    def __init__(self, tfidf_model_struct, model, training_doc_word_pairs=None, training_labels=None):
        # Topic model mush have a tfidf model struct
        self.tfidf_model_struct = tfidf_model_struct
        self.model = model

    def train(self):
        # model.fit(traning_data, label)
        pass

    @classmethod
    def get_model(cls, tfidf_model_struct=None, data_store=None, regen=False):
        md_file_path = get_md_path()
        if not os.path.isfile(md_file_path) or regen:
            engine_logger.info("Generating topic word logistic regression models")

            # transform topic word data to labeled training data
            training_doc_set = []
            training_doc_word_pairs = []
            training_labels = []

            for pair in data_store.topic_word_docs:
                doc = pair[0]
                training_doc_set.append(doc)

                # iterate the combination of doc-word pairs and as positive and negative ex.
                tokens = TfIdfModelStruct.pre_process_doc_tf_idf(doc)
                positive_topics_words = set(p_stemmer.stem(w.lower()) for w in pair[1])
                for t in tokens:
                    training_doc_word_pairs.append((doc, t))
                    if t in positive_topics_words:
                        training_labels.append(1)
                    else:
                        training_labels.append(0)

            # If training the topic model, the tf-idf need to be computed on the training doc set only
            # Not on the whole doc set.
            local_data_store = DataStore({})
            local_data_store.doc_set = training_doc_set
            tfidf_model_struct = TfIdfModelStruct.get_model(data_store=local_data_store, regen=True, save=False)

            # un-trained model
            model = LogisticRegression()
            instance = TopicWordModelStruct(tfidf_model_struct,
                                            model,
                                            training_doc_word_pairs=training_doc_word_pairs,
                                            training_labels=training_labels)
            instance.train()

            # saving
            joblib.dump(instance.model, md_file_path)

            return instance
        else:
            # This tfidf model is loaded outside - i.e. on the whole doc space
            engine_logger.info("Loading existing TF_IDF models.")
            model = joblib.load(md_file_path)

            return TopicWordModelStruct(tfidf_model_struct, model)

    def extract_feature(self, doc_word_pair):
        """
        features:

        0 TF: Term frequency of w in the short text
        1 IDF: Inverse document frequency of w in the whole collection
        2 SF: Number of sentences in the short text that contain w
        3 First: Whether w exists in the first sentence
        4 Last: Whether w exists in the last sentence
        5 NE: Whether w is a named entity (NE)
        6 NE: First Whether w is NE in the first sentence
        7 NE: Last Whether w is NE in the last sentence
        8 POS: Part of speech of w

        :param doc_word_pair:
        :return:
        """
        doc = doc_word_pair[0]
        word = doc_word_pair[1]
        tf, idf = self.tfidf_model_struct.get_tf_and_idf(doc, word)



# # learn LR model
#
# # Logistic Regression
#
# from sklearn import metrics
#
#
# data = [
#     [0.1, 0.2, 0.3, 0.4],
#     [0.9, 0.8, 0.4, 0.5],
#     [0.4, 0.2, 0.1, 0.0],
#     [-0.3, -2.1, -3.2, -3.4],
# ]
# label = [1, 1, 0, 0]
# # fit a logistic regression model to the data
# model = LogisticRegression()
# model.fit(data, label)
#
# # # make predictions
# test = [-0.5, 0.25, 0.35, 0.5]
# expected = 1
# predicted = model.predict(test)
# print predicted
# print model.predict_proba(test)
# # # summarize the fit of the model
# print(metrics.classification_report([expected], predicted))
# print(metrics.confusion_matrix([expected], predicted))
#
# joblib.dump(model, MODEL_FILE_PATH)
#
# new_model = joblib.load(MODEL_FILE_PATH)
#
# print model.predict_proba(test)
