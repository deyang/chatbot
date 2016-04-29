import unittest
from ir_query_engine.query_engine import RankBasedQueryEngineComponent, QueryState, TfIdfModelStruct, LdaModelStruct, \
    TopicWordLookupModelStruct, Response
from ir_query_engine.ranker.ranking import Matcher, MatchFeatures
from ir_query_engine.common import DataStore
from mock import patch

__author__ = 'Deyang'


class QueryEngineTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = [
            {
                "_question": "What do you like to eat?",
                "answer": "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother.",
                "context": [{"Object": "Brocolli"}],
            },
            {
                "_question": "Tell me about your mother.",
                "answer": "My mother spends a lot of time driving my brother around to baseball practice.",
                "context": [{"Person": "Mother"}],
            },
            {
                "_question": "Is driving safe?",
                "answer": "Some health experts suggest that driving may cause increased tension and blood pressure.",
                "context": [{"Concept": "Driving"}],
            },
            {
                "_question": "Does your mother love you?",
                "answer": "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better.",
                "context": [{"Person": "Mother"}],
            },
            {
                "_question": "Want some brocolli?",
                "answer": "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother.",
                "context": [{"Object": "Brocolli"}],
            }
        ]
        cls.data_store = DataStore(cls.data)
        cls.qe = RankBasedQueryEngineComponent(cls.data_store)

    @patch.object(TopicWordLookupModelStruct, 'query', return_value=[(7, 1.0), (1, 3.0)])
    @patch.object(LdaModelStruct, 'query', return_value=[(7, 2.0)])
    @patch.object(TfIdfModelStruct, 'query_questions', return_value=[(1, 1.0), (2, 0.5)])
    def test_retrieve_candidates(self, mock_query_tfidf, mock_query_lda, mock_query_topic):
        qs = QueryState("")
        self.qe._retrieve_candidates(qs)
        self.assertEqual(set(qs.candidate_pairs), set([(4, 5), (6, 7), (2, 3), (0, 1), (8, 1)]))

    def test_retrieve_candidates_from_query_doc_results(self):
        query_results = [(7, 1.0), (1, 3.0), (2, 0.4)]
        qa_pair_candidates = set()
        self.qe._retrieve_candidates_from_query_doc_results(query_results, qa_pair_candidates)
        self.assertEqual(qa_pair_candidates, set([(6, 7), (2, 3), (0, 1), (8, 1)]))

    def test_retrieve_candidates_from_query_question_results(self):
        query_results = [(0, 1.0), (1, 3.0), (2, 0.4)]
        qa_pair_candidates = set()
        self.qe._retrieve_candidates_from_query_question_results(query_results, qa_pair_candidates)
        self.assertEqual(qa_pair_candidates, set([(4, 5), (2, 3), (0, 1)]))

    @patch.object(Matcher, 'match')
    def test_match_candidates(self, mock_match):
        mock_match.return_value = [MatchFeatures(), MatchFeatures()]
        qs = QueryState("query")
        qs.candidate_pairs = [(0, 1), (4, 5)]

        self.qe._match_candidates(qs)

        mock_match.assert_called_with(
            "query",
            [("What do you like to eat?",
              "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."),
             ("Is driving safe?",
              "Some health experts suggest that driving may cause increased tension and blood pressure.")
            ]
        )
        self.assertEqual(len(qs.match_features), 2)
        self.assertIsInstance(qs.match_features[0], MatchFeatures)
        self.assertIsInstance(qs.match_features[1], MatchFeatures)

    @patch.object(RankBasedQueryEngineComponent, '_match_candidates')
    @patch.object(RankBasedQueryEngineComponent, '_retrieve_candidates')
    def test_execute_query(self, mock_retrieve, mock_match):
        def mock_retrieve_side_effect(qs):
            qs.candidate_pairs = [(4, 5), (6, 7), (2, 3), (0, 1), (8, 1)]
        mock_retrieve.side_effect = mock_retrieve_side_effect

        def mock_match_side_effect(qs):
            qs.match_features = [MatchFeatures(question_tf_idf_sim=0.9),
                                 MatchFeatures(question_lda_sim=0.8),
                                 MatchFeatures(answer_lda_sim=0.8),
                                 MatchFeatures(question_topic_word_sim=0.8),
                                 MatchFeatures(question_word2vec_sim=0.8)]

        mock_match.side_effect = mock_match_side_effect

        class MockRankModel(object):

            def __init__(self):
                self.idx = 0
                self.scores = range(5)

            def predict_score(self, f):
                score = self.scores[self.idx]
                self.idx += 1
                return score

        self.qe.rank_model = MockRankModel()

        resp = self.qe.execute_query("test query")
        self.assertIsInstance(resp, Response)
        self.assertEqual(resp.question,
                         "Want some brocolli?")
        self.assertEqual(resp.answer,
                         "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother.")
        self.assertEqual(resp.match_score, 4.0)
        self.assertEqual(resp.feature,
                         MatchFeatures(question_word2vec_sim=0.8).to_vec())
        self.assertEqual(resp.context, [{"Object": "Brocolli"}])






if __name__ == '__main__':
    unittest.main()
