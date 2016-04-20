import unittest
from ir_query_engine.query_engine import QueryEngine, QueryState, TfIdfModelStruct, LdaModelStruct
from ir_query_engine.common import DataStore
from mock import patch

__author__ = 'Deyang'


class QueryEngineTestCase(unittest.TestCase):
    def setUp(self):
        self.data = [
            {
                "question": "What do you like to eat?",
                "answer": "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother.",
                "ranked_answers": []
            },
            {
                "question": "Tell me about your mother.",
                "answer": "My mother spends a lot of time driving my brother around to baseball practice.",
                "ranked_answers": []
            },
            {
                "question": "Is driving safe?",
                "answer": "Some health experts suggest that driving may cause increased tension and blood pressure.",
                "ranked_answers": []
            },
            {
                "question": "Does your mother love you?",
                "answer": "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better.",
                "ranked_answers": []
            },
            {
                "question": "Want some brocolli?",
                "answer": "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother.",
                "ranked_answers": []
            }
        ]
        self.data_store = DataStore(self.data)

    @patch.object(LdaModelStruct, 'query', return_value=[(7, 2.0)])
    @patch.object(TfIdfModelStruct, 'query_answers', return_value=[(0, 2.0)])
    @patch.object(TfIdfModelStruct, 'query_questions', return_value=[(1, 1.0), (2, 0.5)])
    def test_retrieve_candidates(self, mock_query_questions, mock_query_answers, mock_query):
        qs = QueryState("")
        qe = QueryEngine(self.data_store, 1)
        qe._retrieve_candidates(qs)
        self.assertEqual(set(qs.candidate_pairs), set(self.data_store.qid_to_qa_pair.values()))


if __name__ == '__main__':
    unittest.main()
