from ir_query_engine import engine_logger
from ir_query_engine.rank_match_models.simple_features import get_lcs_length, get_edit_distance
import os
from numpy import dot, array
from numpy.linalg import norm
from math import exp

from utils.util import StopWatch
import subprocess

__author__ = 'Deyang'

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_FILE_PATH = os.path.join(DIR_PATH, 'svm_rank', 'data', 'train.dat')
MD_FILE_PATH = os.path.join(DIR_PATH, 'svm_rank', 'data', 'model')
LEARN_BIN_PATH = os.path.join(DIR_PATH, 'svm_rank', 'svm_rank_learn_mac')


def get_train_data_path():
    return TRAIN_DATA_FILE_PATH


def get_md_path():
    return MD_FILE_PATH


def get_bin_path():
    return LEARN_BIN_PATH


class MatchFeatures(object):

    # list of attribute name, to gaurantee the order
    ATTRIBUTE_NAMES = ['question_tfidf_sim',
                       'question_lda_sim',
                       'answer_lda_sim',
                       'question_topic_word_sim',
                       'answer_topic_word_sim',
                       'question_word2vec_sim',
                       'answer_word2vec_sim',
                       'answer_lcs_len',
                       'question_cooccur_size',
                       'answer_cooccur_size',
                       'question_cooccur_rate',
                       'answer_cooccur_rate',
                       'question_cooccur_sum_idf',
                       'answer_cooccur_sum_idf',
                       'question_cooccur_avg_idf',
                       'answer_cooccur_avg_idf',
                       'question_edit_distance'
                       ]

    ATTRIBUTE_ENABLE_DICT = {
        'question_tfidf_sim': 1,
        'question_lda_sim': 1,
        'answer_lda_sim': 1,
        'question_topic_word_sim': 1,
        'answer_topic_word_sim': 1,
        'question_word2vec_sim': 1,
        'answer_word2vec_sim': 1,
        'answer_lcs_len': 1,
        'question_cooccur_size': 1,
        'answer_cooccur_size': 1,
        'question_cooccur_rate': 1,
        'answer_cooccur_rate': 1,
        'question_cooccur_sum_idf': 1,
        'answer_cooccur_sum_idf': 1,
        'question_cooccur_avg_idf': 1,
        'answer_cooccur_avg_idf': 1,
        'question_edit_distance': 1
    }

    def __init__(self,
                 question_tfidf_sim=None,
                 question_lda_sim=None,
                 answer_lda_sim=None,
                 question_topic_word_sim=None,
                 answer_topic_word_sim=None,
                 question_word2vec_sim=None,
                 answer_word2vec_sim=None,
                 answer_lcs_len=None,
                 question_cooccur_size=None,
                 answer_cooccur_size=None,
                 question_cooccur_rate=None,
                 answer_cooccur_rate=None,
                 question_cooccur_sum_idf=None,
                 answer_cooccur_sum_idf=None,
                 question_cooccur_avg_idf=None,
                 answer_cooccur_avg_idf=None,
                 question_edit_distance=None):
        self.question_tfidf_sim = question_tfidf_sim
        self.question_lda_sim = question_lda_sim
        self.answer_lda_sim = answer_lda_sim
        self.question_topic_word_sim = question_topic_word_sim
        self.answer_topic_word_sim = answer_topic_word_sim
        self.question_word2vec_sim = question_word2vec_sim
        self.answer_word2vec_sim = answer_word2vec_sim
        self.answer_lcs_len = answer_lcs_len
        self.question_cooccur_size = question_cooccur_size
        self.answer_cooccur_size = answer_cooccur_size
        self.question_cooccur_rate = question_cooccur_rate
        self.answer_cooccur_rate = answer_cooccur_rate
        self.question_cooccur_sum_idf = question_cooccur_sum_idf
        self.answer_cooccur_sum_idf = answer_cooccur_sum_idf
        self.question_cooccur_avg_idf = question_cooccur_avg_idf
        self.answer_cooccur_avg_idf = answer_cooccur_avg_idf
        self.question_edit_distance = question_edit_distance

    def to_vec(self):
        vec = []
        for attr_name in self.ATTRIBUTE_NAMES:
            if self.ATTRIBUTE_ENABLE_DICT[attr_name]:
                vec.append(getattr(self, attr_name))
        return vec

    def __str__(self):
        vec = self.to_vec()
        splits = []
        for idx, val in enumerate(vec):
            splits.append("%s:%s" % (idx + 1, val))
        return " ".join(splits)

    def get_num_features(self):
        """ Get the number of enabled features"""
        return sum(self.ATTRIBUTE_ENABLE_DICT.values())


class Matcher(object):
    """
    Apply all the match models

    """
    def __init__(self, tfidf_model_struct, lda_model_struct, topic_word_model_struct, word2vec_model):
        self.tfidf_model_struct = tfidf_model_struct
        self.lda_model_struct = lda_model_struct
        self.topic_word_model_struct = topic_word_model_struct
        self.word2vec_model = word2vec_model

    def match(self, query_doc, compare_question_answer_pairs, match_results=None):
        question_docs = [pair[0] for pair in compare_question_answer_pairs]
        answer_docs = [pair[1] for pair in compare_question_answer_pairs]
        if match_results is None:
            match_results = list()
            for _ in range(len(compare_question_answer_pairs)):
                match_results.append(MatchFeatures())

        # stopwatch = StopWatch()
        # match question tfidf
        if MatchFeatures.ATTRIBUTE_ENABLE_DICT['question_tfidf_sim']:
            sims = self.tfidf_model_struct.get_similarities(query_doc,
                                                            question_docs)
            for idx, sim in sims:
                match_results[idx].question_tfidf_sim = sim

        if MatchFeatures.ATTRIBUTE_ENABLE_DICT['question_lda_sim']:
            # match question lda
            sims = self.lda_model_struct.get_similarities(query_doc,
                                                          question_docs)
            for idx, sim in sims:
                match_results[idx].question_lda_sim = sim

        if MatchFeatures.ATTRIBUTE_ENABLE_DICT['answer_lda_sim']:
            # match answer lda
            sims = self.lda_model_struct.get_similarities(query_doc,
                                                          answer_docs)
            for idx, sim in sims:
                match_results[idx].answer_lda_sim = sim

        # print "after lda: %s" % sw.lap()
        if MatchFeatures.ATTRIBUTE_ENABLE_DICT['question_topic_word_sim']:
            # match question topic words
            sims = self.topic_word_model_struct.get_similarities(query_doc,
                                                                 question_docs)
            for idx, sim in sims:
                match_results[idx].question_topic_word_sim = sim

        if MatchFeatures.ATTRIBUTE_ENABLE_DICT['answer_topic_word_sim']:
            # match answer topic words
            sims = self.topic_word_model_struct.get_similarities(query_doc,
                                                                 answer_docs)
            for idx, sim in sims:
                match_results[idx].answer_topic_word_sim = sim

        if MatchFeatures.ATTRIBUTE_ENABLE_DICT['question_word2vec_sim']:
            # match question word2vec
            sims = self.word2vec_model.get_similarities(query_doc,
                                                        question_docs)
            for idx, sim in sims:
                match_results[idx].question_word2vec_sim = sim

        if MatchFeatures.ATTRIBUTE_ENABLE_DICT['answer_word2vec_sim']:
            # match answer word2vec
            sims = self.word2vec_model.get_similarities(query_doc,
                                                        answer_docs)
            for idx, sim in sims:
                match_results[idx].answer_word2vec_sim = sim

        # print "sims: %s" % sw.stop()
        # match lcs between the answer
        for idx, answer_doc in enumerate(answer_docs):
            match_results[idx].answer_lcs_len = get_lcs_length(query_doc, answer_doc)

        # match cooccurance between the question
        for idx, question_doc in enumerate(question_docs):
            coocur_size, coocur_rate, cooccur_sum_idf, cooccur_avg_idf = \
                self.tfidf_model_struct.get_cooccur_features(query_doc, question_doc)
            match_results[idx].question_cooccur_size = coocur_size
            match_results[idx].question_cooccur_rate = coocur_rate
            match_results[idx].question_cooccur_sum_idf = cooccur_sum_idf
            match_results[idx].question_cooccur_avg_idf = cooccur_avg_idf

        # match cooccurance between the answers
        for idx, answer_doc in enumerate(answer_docs):
            coocur_size, coocur_rate, cooccur_sum_idf, cooccur_avg_idf = \
                self.tfidf_model_struct.get_cooccur_features(query_doc, answer_doc)
            match_results[idx].answer_cooccur_size = coocur_size
            match_results[idx].answer_cooccur_rate = coocur_rate
            match_results[idx].answer_cooccur_sum_idf = cooccur_sum_idf
            match_results[idx].answer_cooccur_avg_idf = cooccur_avg_idf

        # sw.restart()
        # match edit dist between the questions
        for idx, question_doc in enumerate(question_docs):
            match_results[idx].question_edit_distance = get_edit_distance(query_doc, question_doc)
        # print "edit dis: %s" % sw.stop()

        # print "total match: %s" % stopwatch.stop()
        return match_results


class RankTrainingDataGenerator(object):

    def __init__(self, matcher, rank_data, query_id_offset=0):
        self.matcher = matcher
        self.rank_data = rank_data
        self.query_id_offset = query_id_offset

    def write_training_data(self):
        query_id = self.query_id_offset
        engine_logger.info("Writing training data. Num of queries: %s" % len(self.rank_data))
        file_path = "%s.offset%s" % (get_train_data_path(), self.query_id_offset)
        with open(file_path, 'w') as f:
            for question, pairs in self.rank_data:
                engine_logger.debug("Writing for query %s" % query_id)
                f.write("# query %s\n" % query_id)
                qa_pairs = [t[0] for t in pairs]
                features = self.matcher.match(question, qa_pairs)
                for idx, feature in enumerate(features):
                    labeled_score = pairs[idx][1]
                    f.write("%s qid:%s %s\n" % (labeled_score, query_id, str(feature)))

                query_id += 1


class RankModelTrainer(object):
    RANK_SVM_KERNEL_TYPE_LINEAR = 'linear'
    RANK_SVM_KERNEL_TYPE_RBF = 'rbf'

    def __init__(self, svm_kernel_type, c, gamma=0.5):
        engine_logger.info("Initializeing RankModelTrainer. Kernel type: %s, c: %s, gamma: %s" %
                           (svm_kernel_type, c, gamma))
        self.svm_kernel_type = svm_kernel_type
        self.c = c
        self.gamma = gamma

    def train_model(self):
        bin_path = get_bin_path()
        train_data_path = get_train_data_path()
        model_path = get_md_path()
        if self.svm_kernel_type == self.RANK_SVM_KERNEL_TYPE_LINEAR:
            subprocess.call([bin_path, '-c', str(self.c), train_data_path, model_path])
            engine_logger.info("Finished training linear SVM model")
        elif self.svm_kernel_type == self.RANK_SVM_KERNEL_TYPE_RBF:
            subprocess.call(
                [bin_path, '-c', str(self.c), '-t', '2', '-g', str(self.gamma), train_data_path, model_path])
            engine_logger.info("Finished RBF kernel based SVM model")


def read_rank_model_from_file():
    md_file_path = get_md_path()
    if not os.path.isfile(md_file_path):
        raise Exception("Missing model file: %s" % md_file_path)

    with open(md_file_path, 'r') as f:
        contents = f.readlines()
        type_line = contents[1]
        type_line_splits = type_line.split(" ")
        if type_line_splits[0] == "0":
            # kernel type: linear
            threshold_line = contents[-2]
            threshold = float(threshold_line.split(" ")[0])
            sv_line = contents[-1]
            splits = sv_line.split(" ")[1:-1]
            weight_vec = []
            for part in splits:
                weight_vec.append(float(part.split(":")[1]))

            engine_logger.info("Load linear SVM rank model from file")
            return LinearSVMRankModel(weight_vec=weight_vec,
                                      threshold=threshold)

        elif type_line_splits[0] == "2":
            # kernel type: rbf
            gamma_param_line = contents[3]
            gamma = float(gamma_param_line.split(" ")[0])
            threshold_line = contents[10]
            threshold = float(threshold_line.split(" ")[0])
            alphays = []
            svs = []
            for line in contents[11:]:
                line_splits = line.split(" ")
                alphays.append(float(line_splits[0]))
                sv_vec = []
                for part in line_splits[1:-1]:
                    sv_vec.append(float(part.split(":")[1]))
                svs.append(sv_vec)

            engine_logger.info("Load RBF kernel SVM rank model from file")
            return RBFSVMRankModel(alphays, svs, gamma, threshold)
        else:
            engine_logger.error("Un-recognized model file format")


class RBFSVMRankModel(object):

    def __init__(self, alphays, svs, gamma, threshold):
        self.alphays = alphays
        self.svs = svs
        self.gamma = gamma
        self.threshold = threshold

    def predict_score(self, features):
        if isinstance(features, MatchFeatures):
            features = features.to_vec()

        summation = 0.0
        for alphay, sv_vec in zip(self.alphays, self.svs):
            summation += alphay * self._kernel_func(features, sv_vec)
        return summation - self.threshold

    def _kernel_func(self, vec_a, vec_b):
        dist = array(vec_a) - array(vec_b)
        n = norm(dist)
        return exp(-1 * self.gamma * n * n)


class LinearSVMRankModel(object):

    def __init__(self, weight_vec=None, threshold=None):
        self.weight_vec = weight_vec
        self.threshold = threshold

    def predict_score(self, features):
        if isinstance(features, MatchFeatures):
            features = features.to_vec()

        return dot(self.weight_vec, features) - self.threshold
