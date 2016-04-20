from ir_query_engine import engine_logger
from ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model import TfIdfModelStruct
from ir_query_engine.retrieve_match_models.lda_feature.lda_model import LdaModelStruct
from ir_query_engine.rank_match_models.topic_word_feature.topic_word_model import TopicWordModelStruct
from ir_query_engine.rank_match_models.word2vec_feature.word2vec_model import Word2VecModel


__author__ = 'Deyang'


class QueryState(object):

    def __init__(self, raw_query):
        self.raw_query = raw_query
        self.candidate_pairs = []


class QueryEngine(object):

    def __init__(self, data_store, num_topics):
        self.data_store = data_store
        self.num_topics = num_topics
        self.tfidf_model_struct = TfIdfModelStruct.get_model()
        self.lda_model_struct = LdaModelStruct.get_model(num_topics=self.num_topics)
        self.topic_word_model_struct = TopicWordModelStruct.get_model(tfidf_model_struct=self.tfidf_model_struct)
        self.word2vec_model = Word2VecModel()

    def execute_query(self, raw_query):
        query_state = QueryState(raw_query)
        engine_logger.info("State one, retrieving candidates.")
        self._retrieve_candidates(query_state)

    def _retrieve_candidates(self, query_state):
        # retrieve similar questions based on tf-idf
        results = self.tfidf_model_struct.query_questions(raw_doc=query_state.raw_query)
        results = self.data_store.translate_question_query_results(results)
        qa_pairs = set()
        for qid, _ in results:
            qa_pairs.add(self.data_store.qid_to_qa_pair[qid])
        engine_logger.info("Candidates after tf-idf question matching: %s" % qa_pairs)

        # retrieve similar answers based on tf-idf
        results = self.tfidf_model_struct.query_answers(raw_doc=query_state.raw_query)
        results = self.data_store.translate_answer_query_results(results)

        for aid, _ in results:
            qa_pairs.update(self.data_store.aid_to_qa_pairs[aid])
        engine_logger.info("Candidates after tf-idf answer matching: %s" % qa_pairs)

        # retrieve similar docs (question or answer) based on lda
        results = self.lda_model_struct.query(raw_doc=query_state.raw_query)
        for doc_id, _ in results:
            if doc_id in self.data_store.question_set:
                qa_pairs.add(self.data_store.qid_to_qa_pair[doc_id])
            elif doc_id in self.data_store.answer_set:
                qa_pairs.update(self.data_store.aid_to_qa_pairs[doc_id])
        engine_logger.info("Candidates after lda matching: %s" % qa_pairs)

        query_state.candidate_pairs = list(qa_pairs)

    def _match_candidates(self, query_state):
        # apply all the match models to all the candidates
        pass



