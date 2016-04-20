from ir_query_engine.rank_match_models.simple_features import get_lcs_length
__author__ = 'Deyang'


class MatchFeatures(object):

    NUM_OF_FEATURES = 16

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
               self.answer_word2vec_sim, self.lcs_len, self.question_cooccur_rate, self.answer_cooccur_rate, \
               self.question_cooccur_sum_idf, self.answer_cooccur_sum_idf, self.question_cooccur_avg_idf, \
               self.answer_cooccur_avg_idf, self.question_edit_distance


class Matcher(object):
    """
    Apply allt the match models

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
        for idx, sim in enumerate(sims):
            match_results[idx].question_tf_idf_sim = sim[1]

        # match answer tfidf
        sims = self.tfidf_model_struct.get_similarities(query_doc,
                                                        answer_docs)
        for idx, sim in enumerate(sims):
            match_results[idx].answer_tf_idf_sim = sim[1]

        # match question lda
        sims = self.lda_model_struct.get_similarities(query_doc,
                                                      question_docs)
        for idx, sim in enumerate(sims):
            match_results[idx].question_lda_sim = sim[1]

        # match answer lda
        sims = self.lda_model_struct.get_similarities(query_doc,
                                                      answer_docs)
        for idx, sim in enumerate(sims):
            match_results[idx].answer_lda_sim = sim[1]

        # match question topic words
        sims = self.topic_word_model_struct.get_similarities(query_doc,
                                                             question_docs)
        for idx, sim in enumerate(sims):
            match_results[idx].question_topic_word_sim = sim[1]

        # match answer topic words
        sims = self.topic_word_model_struct.get_similarities(query_doc,
                                                             answer_docs)
        for idx, sim in enumerate(sims):
            match_results[idx].answer_topic_word_sim = sim[1]

        # match question word2vec
        sims = self.word2vec_model.get_similarities(query_doc,
                                                    question_docs)
        for idx, sim in enumerate(sims):
            match_results[idx].question_word2vec_sim = sim[1]

        # match answer word2vec
        sims = self.word2vec_model.get_similarities(query_doc,
                                                    answer_docs)
        for idx, sim in enumerate(sims):
            match_results[idx].answer_word2vec_sim = sim[1]

        # match lcs between the answer
        for ids, answer_doc in enumerate(answer_docs):
            match_results[idx].answer_lcs_len = get_lcs_length(query_doc, answer_doc)


