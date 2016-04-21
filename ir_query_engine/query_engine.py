from ir_query_engine import engine_logger
from ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model import TfIdfModelStruct
from ir_query_engine.retrieve_match_models.lda_feature.lda_model import LdaModelStruct
from ir_query_engine.rank_match_models.topic_word_feature.topic_word_model import TopicWordModelStruct
from ir_query_engine.rank_match_models.word2vec_feature.word2vec_model import Word2VecModel
from ir_query_engine.ranker.ranking import Matcher, LinearRankModel

__author__ = 'Deyang'


class QueryState(object):

    def __init__(self, raw_query):
        self.raw_query = raw_query
        self.candidate_pairs = []
        self.match_features = []
        self.candidate_scores = []
        self.responses = []
        self.response_doc = None


class QueryEngine(object):

    def __init__(self, data_store, num_topics):
        self.data_store = data_store
        self.num_topics = num_topics
        self.tfidf_model_struct = TfIdfModelStruct.get_model()
        self.lda_model_struct = LdaModelStruct.get_model()
        self.topic_word_model_struct = TopicWordModelStruct.get_model(tfidf_model_struct=self.tfidf_model_struct)
        self.word2vec_model = Word2VecModel()
        self.matcher = Matcher(
            self.tfidf_model_struct,
            self.lda_model_struct,
            self.topic_word_model_struct,
            self.word2vec_model
        )
        self.rank_model = LinearRankModel.read_model_from_file()

        engine_logger.info("Query engine is up")

    def execute_query(self, raw_query):
        engine_logger.info("Raw query: %s" % raw_query)
        query_state = QueryState(raw_query)
        engine_logger.info("State one, retrieving candidates.")
        self._retrieve_candidates(query_state)
        engine_logger.info("State two, matching candidates.")
        self._match_candidates(query_state)

        # get the scores
        for feature in query_state.match_features:
            query_state.candidate_scores.append(
                self.rank_model.predict_score(feature)
            )

        query_state.responses = zip(query_state.candidate_pairs, query_state.candidate_scores)
        engine_logger.info("State three, ranking candidates.")
        query_state.responses.sort(key=lambda pair: pair[1], reverse=True)

        top5 = []
        for pair in query_state.responses[0:5]:
            top5.append(
                (
                    self.data_store.doc_set[pair[0][0]],
                    self.data_store.doc_set[pair[0][1]],
                    pair[1]
                )
            )

        engine_logger.info("Ranked top 5 responses: %s" % top5)
        query_state.response_doc = self.data_store.doc_set[query_state.responses[0][0][1]]
        return query_state.response_doc

    def _retrieve_candidates(self, query_state):
        # retrieve similar questions based on tf-idf
        results = self.tfidf_model_struct.query_questions(raw_doc=query_state.raw_query)
        results = self.data_store.translate_question_query_results(results)
        qa_pairs = set()

        qa_pairs_from_question_tfidf = []
        for qid, sim in results:
            qa_pair = self.data_store.qid_to_qa_pair[qid]
            engine_logger.debug("Pair from question tfidf matching: %s, sim: %s" %
                                (self.data_store.get_docs_by_pair(qa_pair), sim))
            qa_pairs_from_question_tfidf.append(qa_pair)
        qa_pairs.update(qa_pairs_from_question_tfidf)
        engine_logger.debug("Candidates from tf-idf question matching: %s" % qa_pairs_from_question_tfidf)

        # retrieve similar answers based on tf-idf
        results = self.tfidf_model_struct.query_answers(raw_doc=query_state.raw_query)
        results = self.data_store.translate_answer_query_results(results)

        qa_pairs_from_answer_tfidf = []
        for aid, sim in results:
            pairs = self.data_store.aid_to_qa_pairs[aid]
            for pair in pairs:
                engine_logger.debug("Pair from answer tfidf matching: %s, sim: %s" %
                                    (self.data_store.get_docs_by_pair(pair), sim))
            qa_pairs_from_answer_tfidf.extend(pairs)
        qa_pairs.update(qa_pairs_from_answer_tfidf)
        engine_logger.debug("Candidates after tf-idf answer matching: %s" % qa_pairs_from_answer_tfidf)

        # retrieve similar docs (question or answer) based on lda
        results = self.lda_model_struct.query(raw_doc=query_state.raw_query)

        qa_pairs_from_lda = []
        for doc_id, sim in results:
            if doc_id in self.data_store.question_set:
                pair = self.data_store.qid_to_qa_pair[doc_id]
                qa_pairs_from_lda.append(pair)
                engine_logger.debug("Pair from question lda matching: %s, sim: %s" %
                                    (self.data_store.get_docs_by_pair(qa_pair), sim))
            elif doc_id in self.data_store.answer_set:
                pairs = self.data_store.aid_to_qa_pairs[doc_id]
                for pair in pairs:
                    engine_logger.debug("Pair from answer lda matching: %s, sim: %s" %
                                        (self.data_store.get_docs_by_pair(pair), sim))
                qa_pairs_from_lda.extend(pairs)
            else:
                engine_logger.debug("Matched non question not answer doc from lda: %s" % self.data_store.doc_set[doc_id])
        qa_pairs.update(qa_pairs_from_lda)
        engine_logger.debug("Candidates after lda matching: %s" % qa_pairs_from_lda)

        query_state.candidate_pairs = list(qa_pairs)
        engine_logger.info("Candidates: %s, len: %s" % (query_state.candidate_pairs, len(query_state.candidate_pairs)))

    def _match_candidates(self, query_state):
        # apply all the match models to all the candidates
        question_answer_pairs = [(self.data_store.doc_set[qid],
                                  self.data_store.doc_set[aid])
                                 for qid, aid in query_state.candidate_pairs]

        query_state.match_features = self.matcher.match(query_state.raw_query,
                                                        question_answer_pairs)
