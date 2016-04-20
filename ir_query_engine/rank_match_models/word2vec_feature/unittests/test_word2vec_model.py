import unittest
from ir_query_engine.rank_match_models.word2vec_feature.word2vec_model import Word2VecModel

__author__ = 'Deyang'


class Word2VecModelTestCase(unittest.TestCase):

    def test_get_similarities(self):
        doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
        doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
        doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
        doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
        doc_e = "Health professionals say that brocolli is good for your health."

        doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]
        q = "Is brocolli tasty to eat?"

        m = Word2VecModel()

        results = m.get_similarities(q, doc_set)
        self.assertEqual(results[0][0], 0)


if __name__ == '__main__':
    unittest.main()
