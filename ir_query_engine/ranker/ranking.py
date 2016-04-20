from ir_query_engine.rank_match_models.simple_features import get_lcs_length, get_edit_distance
__author__ = 'Deyang'


class MatchFeatures(object):

    NUM_OF_FEATURES = 18

    def __init__(self,
                 question_tf_idf_sim=None,
                 answer_tf_idf_sim=None,
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

        self.question_tf_idf_sim = question_tf_idf_sim
        self.answer_tf_idf_sim = answer_tf_idf_sim
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
        return self.question_tf_idf_sim, self.answer_tf_idf_sim, self.question_lda_sim, self.answer_lda_sim,\
               self.question_topic_word_sim, self.answer_topic_word_sim, self.question_word2vec_sim, \
               self.answer_word2vec_sim, self.answer_lcs_len, self.question_cooccur_size, self.answer_cooccur_size, \
               self.question_cooccur_rate, self.answer_cooccur_rate, self.question_cooccur_sum_idf, \
               self.answer_cooccur_sum_idf, self.question_cooccur_avg_idf, \
               self.answer_cooccur_avg_idf, self.question_edit_distance


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

        # match question tfidf
        sims = self.tfidf_model_struct.get_similarities(query_doc,
                                                        question_docs)
        for idx, sim in sims:
            match_results[idx].question_tf_idf_sim = sim

        # match answer tfidf
        sims = self.tfidf_model_struct.get_similarities(query_doc,
                                                        answer_docs)
        for idx, sim in sims:
            match_results[idx].answer_tf_idf_sim = sim

        # match question lda
        sims = self.lda_model_struct.get_similarities(query_doc,
                                                      question_docs)
        for idx, sim in sims:
            match_results[idx].question_lda_sim = sim

        # match answer lda
        sims = self.lda_model_struct.get_similarities(query_doc,
                                                      answer_docs)
        for idx, sim in sims:
            match_results[idx].answer_lda_sim = sim

        # match question topic words
        sims = self.topic_word_model_struct.get_similarities(query_doc,
                                                             question_docs)
        for idx, sim in sims:
            match_results[idx].question_topic_word_sim = sim

        # match answer topic words
        sims = self.topic_word_model_struct.get_similarities(query_doc,
                                                             answer_docs)
        for idx, sim in sims:
            match_results[idx].answer_topic_word_sim = sim

        # match question word2vec
        sims = self.word2vec_model.get_similarities(query_doc,
                                                    question_docs)
        for idx, sim in sims:
            match_results[idx].question_word2vec_sim = sim

        # match answer word2vec
        sims = self.word2vec_model.get_similarities(query_doc,
                                                    answer_docs)
        for idx, sim in sims:
            match_results[idx].answer_word2vec_sim = sim

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

        # match edit dist between the questions
        for idx, question_doc in enumerate(question_docs):
            match_results[idx].question_edit_distance = get_edit_distance(query_doc, question_doc)

        return match_results
