from ir_query_engine import engine_logger
from ir_query_engine.match_models.tfidf_model import TfIdfModelStruct
from ir_query_engine.match_models.lda_model import LdaModelStruct
from ir_query_engine.match_models.topic_word_lookup_model import TopicWordLookupModelStruct
from ir_query_engine.match_models.word2vec_model import Word2VecModel
from ir_query_engine.ranker.ranking import Matcher, read_rank_model_from_file
from ir_query_engine.judge.classifier_and_calibration import CalibratedBoostedDecisionStumpsFactory

__author__ = 'Deyang'


class QueryState(object):

    def __init__(self, raw_query):
        self.raw_query = raw_query
        self.candidate_pairs = []
        self.match_features = []
        self.candidate_scores = []
        self.responses = []
        self.final_response = None


class Response(object):

    def __init__(self, question, answer, match_score, feature, context, confidence_score=None):
        self.question = question
        self.answer = answer
        # match score is the original feature match score
        self.match_score = match_score
        self.feature = feature
        self.context = context
        # confidence score is a calibrated probability indicating the confidence
        self.confidence_score = confidence_score

    def __str__(self):
        return "Response<question: %s, answer: %s, match_score: %s, feature: %s, context: %s, confidence_score: %s>" % \
               (self.question, self.answer, self.match_score, self.feature, self.context, self.confidence_score)


class QueryEngineAbstract(object):

    def __init__(self, data_store):
        self.data_store = data_store
        self.is_up = False

    def get_status(self):
        return self.is_up

    def load_models(self):
        raise NotImplementedError

    def set_models(self):
        raise NotImplementedError

    def execute_query(self, raw_query):
        raise NotImplementedError


class CompositeQueryEngine(QueryEngineAbstract):

    def __init__(self, data_store, eager_loading=True):
        super(CompositeQueryEngine, self).__init__(data_store)
        self.query_components = None
        self.final_judge = None
        if eager_loading:
            self.load_models()

    def get_status(self):
        return reduce(lambda ret, comp: ret & comp.get_status(), self.query_components, initial=self.is_up)

    def load_models(self):
        tfidf_query_engine_component = TfIdfQueryEngineComponent(self.data_store)
        rank_based_query_engine_component = RankBasedQueryEngineComponent(self.data_store)
        self.query_components = [tfidf_query_engine_component, rank_based_query_engine_component]
        self.final_judge = CalibratedBoostedDecisionStumpsFactory.load_existing_model()
        self.is_up = True

    def set_models(self, query_components, final_judge):
        self.query_components = query_components
        self.final_judge = final_judge
        self.is_up = True

    def execute_query(self, raw_query):
        responses_from_components = map(lambda component: component.execute_query(raw_query),
                                        self.query_components)
        judge_input = map(lambda resp: resp.match_score, responses_from_components)
        predicts = self.final_judge.predict_proba(judge_input)
        idx, _ = max(enumerate(predicts), key=lambda t: t[1])
        if idx < len(responses_from_components):
            final_resp = responses_from_components[idx]
        else:
            final_resp = Response(
                "",
                "",
                None,
                None,
                None
            )

        final_resp.confidence_score = sum(predicts[:-1])
        return final_resp


class TfIdfQueryEngineComponent(QueryEngineAbstract):

    def __init__(self, data_store, eager_loading=True):
        super(TfIdfQueryEngineComponent, self).__init__(data_store)
        self.tfidf_model = None
        if eager_loading:
            self.load_models()

    def load_models(self):
        self.tfidf_model = TfIdfModelStruct.get_model()
        self.is_up = True

    def set_models(self, tfidf_model):
        self.tfidf_model = tfidf_model
        self.is_up = True

    def execute_query(self, raw_query):
        engine_logger.debug("Execute raw query: %s in tfidf query engine component" % raw_query)
        raw_query = raw_query.lower()
        query_state = QueryState(raw_query)
        results = self.tfidf_model.query_questions(raw_doc=query_state.raw_query)
        results = self.data_store.translate_question_query_results(results)
        top_doc_id, sim = results[0]
        qid, aid = self.data_store.qid_to_qa_pair[top_doc_id]

        return Response(
            self.data_store.doc_set[qid], # question
            self.data_store.doc_set[aid], # answer
            sim,
            None,
            self.data_store.qa_context.get(qid, None) # context
        )


class RankBasedQueryEngineComponent(QueryEngineAbstract):
    """
    Combine multi-models and rank the final results
    """

    def __init__(self, data_store, eager_loading=True):
        super(RankBasedQueryEngineComponent, self).__init__(data_store)
        self.tfidf_model = None
        self.lda_model = None
        self.topic_word_lookup_model = None
        self.word2vec_model = None
        self.matcher = None
        self.rank_model = None
        if eager_loading:
            self.load_models()

    def load_models(self):
        """
        Load all the models from files
        :return
        """
        self.tfidf_model = TfIdfModelStruct.get_model()
        self.lda_model = LdaModelStruct.get_model()
        self.topic_word_lookup_model = TopicWordLookupModelStruct.get_model()
        self.word2vec_model = Word2VecModel()
        self.matcher = Matcher(
            self.tfidf_model,
            self.lda_model,
            self.topic_word_lookup_model,
            self.word2vec_model
        )
        self.rank_model = read_rank_model_from_file()
        engine_logger.info("Rank based query engine is up")
        self.is_up = True

    def set_models(self, tfidf_model, lda_model, topic_word_lookup_model, word2vec_model, rank_model):
        self.tfidf_model = tfidf_model
        self.lda_model = lda_model
        self.topic_word_lookup_model = topic_word_lookup_model
        self.word2vec_model = word2vec_model
        self.rank_model = rank_model
        self.matcher = Matcher(
            self.tfidf_model,
            self.lda_model,
            self.topic_word_lookup_model,
            self.word2vec_model
        )
        self.is_up = True

    def execute_query(self, raw_query):
        engine_logger.debug("Execute raw query: %s in ranked query engine component" % raw_query)
        raw_query = raw_query.lower()
        query_state = QueryState(raw_query)
        engine_logger.debug("State one, retrieving candidates.")
        self._retrieve_candidates(query_state)
        engine_logger.debug("State two, matching candidates.")
        self._match_candidates(query_state)

        # get the scores
        for feature in query_state.match_features:
            query_state.candidate_scores.append(
                self.rank_model.predict_score(feature)
            )

        for idx in range(len(query_state.candidate_pairs)):
            resp = Response(
                self.data_store.doc_set[query_state.candidate_pairs[idx][0]], # question
                self.data_store.doc_set[query_state.candidate_pairs[idx][1]], # answer
                query_state.candidate_scores[idx], # match score
                query_state.match_features[idx].to_vec(), # feature
                self.data_store.qa_context.get(query_state.candidate_pairs[idx][0], None) # context
            )
            query_state.responses.append(resp)

        engine_logger.debug("State three, ranking candidates.")
        query_state.responses.sort(key=lambda r: r.match_score, reverse=True)

        engine_logger.debug("Ranked top 5 responses: %s" % query_state.responses[:5])

        query_state.final_response = query_state.responses[0]
        engine_logger.debug(query_state.final_response)
        return query_state.final_response

    def _retrieve_candidates_from_query_doc_results(self, results, qa_pair_candidates, model_name=""):
        assert isinstance(qa_pair_candidates, set)
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
        assert isinstance(qa_pair_candidates, set)
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
        results = self.tfidf_model.query_questions(raw_doc=query_state.raw_query)
        self._retrieve_candidates_from_query_question_results(results, qa_pair_candidates, model_name="TFIDF")
        # engine_logger.debug("Candidates from tf-idf question matching: %s" % qa_pairs_from_question_tfidf)

        # retrieve similar docs (question or answer) based on lda
        results = self.lda_model.query(raw_doc=query_state.raw_query)
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
