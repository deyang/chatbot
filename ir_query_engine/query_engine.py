from ir_query_engine import engine_logger
from ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model import TfIdfModelStruct
from ir_query_engine.retrieve_match_models.lda_feature.lda_model import LdaModelStruct
from ir_query_engine.rank_match_models.topic_word_lookup_feature.topic_word_lookup_model import TopicWordLookupModelStruct
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
        self.response= None


class Response(object):
    def __init__(self, question, answer, score, feature, context):
        self.question = question
        self.answer = answer
        self.score = score
        self.feature = feature
        self.context = context


class QueryEngine(object):

    def __init__(self, data_store):
        self.data_store = data_store
        self.tfidf_model_struct = TfIdfModelStruct.get_model()
        self.lda_model_struct = LdaModelStruct.get_model()
        self.topic_word_lookup_model = \
            TopicWordLookupModelStruct.get_model()
        self.word2vec_model = Word2VecModel()
        self.matcher = Matcher(
            self.tfidf_model_struct,
            self.lda_model_struct,
            self.topic_word_lookup_model,
            self.word2vec_model
        )
        self.rank_model = LinearRankModel.read_model_from_file()

        engine_logger.info("Query engine is up")

    def execute_query(self, raw_query):
        raw_query = raw_query.lower()
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

        query_state.responses = zip(query_state.candidate_pairs,
                                    query_state.candidate_scores,
                                    query_state.match_features)
        engine_logger.info("State three, ranking candidates.")
        query_state.responses.sort(key=lambda pair: pair[1], reverse=True)

        top5 = []
        for pair in query_state.responses[0:5]:
            top5.append(
                (
                    self.data_store.doc_set[pair[0][0]],
                    self.data_store.doc_set[pair[0][1]],
                    pair[1],
                    pair[2].to_vec()
                )
            )

        engine_logger.info("Ranked top 5 responses: %s" % top5)

        resp = Response(
            self.data_store.doc_set[query_state.responses[0][0][0]], # question
            self.data_store.doc_set[query_state.responses[0][0][1]], # answer
            query_state.responses[0][1], # score
            query_state.responses[0][2].to_vec(), # feature
            self.data_store.qa_context.get(query_state.responses[0][0][0], None)
        )
        query_state.response = resp
        print resp.question
        print resp.feature
        print resp.score
        print resp.context

        return query_state.response

    def _retrieve_candidates_from_query_doc_results(self, results, qa_pair_candidates, model_name=""):
        retrieved_qa_pairs = []
        for doc_id, sim in results:
            if doc_id in self.data_store.question_set:
                pair = self.data_store.qid_to_qa_pair[doc_id]
                retrieved_qa_pairs.append(pair)
                engine_logger.debug("Pair from question matching: model %s, %s, sim: %s" %
                                    (model_name, self.data_store.get_docs_by_pair(pair), sim))
            elif doc_id in self.data_store.answer_set:
                pairs = self.data_store.aid_to_qa_pairs[doc_id]
                retrieved_qa_pairs.extend(pairs)
                for pair in pairs:
                    engine_logger.debug("Pair from answer matching: model %s, %s, sim: %s" %
                                        (model_name, self.data_store.get_docs_by_pair(pair), sim))
            else:
                engine_logger.error("Matched not question nor answer doc from model %s: %s" %
                                    (model_name, self.data_store.doc_set[doc_id]))

        qa_pair_candidates.update(retrieved_qa_pairs)

    def _retrieve_candidates_from_query_question_results(self, results, qa_pair_candidates, model_name=""):
        retrieved_qa_pairs = []
        results = self.data_store.translate_question_query_results(results)
        for qid, sim in results:
            pair = self.data_store.qid_to_qa_pair[qid]
            engine_logger.debug("Pair from question matching: model %s, %s, sim: %s" %
                                (model_name, self.data_store.get_docs_by_pair(pair), sim))
            retrieved_qa_pairs.append(pair)
        qa_pair_candidates.update(retrieved_qa_pairs)

    def _retrieve_candidates(self, query_state):
        qa_pair_candidates = set()

        # retrieve similar questions based on tf-idf
        results = self.tfidf_model_struct.query_questions(raw_doc=query_state.raw_query)
        self._retrieve_candidates_from_query_question_results(results, qa_pair_candidates, model_name="TFIDF")
        # engine_logger.debug("Candidates from tf-idf question matching: %s" % qa_pairs_from_question_tfidf)

        # retrieve similar docs (question or answer) based on lda
        results = self.lda_model_struct.query(raw_doc=query_state.raw_query)
        self._retrieve_candidates_from_query_doc_results(results, qa_pair_candidates, model_name="LDA")
        # engine_logger.debug("Candidates after lda matching: %s" % qa_pairs_from_lda)

        # retrieve similar docs (question or answer) based on topic words lookup
        results = self.topic_word_lookup_model.query(raw_doc=query_state.raw_query)
        self._retrieve_candidates_from_query_doc_results(results, qa_pair_candidates, model_name="TopicWordLookup")

        query_state.candidate_pairs = list(qa_pair_candidates)
        # engine_logger.info("Candidates: %s, len: %s" % (query_state.candidate_pairs, len(query_state.candidate_pairs)))

    def _match_candidates(self, query_state):
        # apply all the match models to all the candidates
        question_answer_pairs = [(self.data_store.doc_set[qid],
                                  self.data_store.doc_set[aid])
                                 for qid, aid in query_state.candidate_pairs]

        query_state.match_features = self.matcher.match(query_state.raw_query,
                                                        question_answer_pairs)
