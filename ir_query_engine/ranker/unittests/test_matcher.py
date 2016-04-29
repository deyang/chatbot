import unittest
from mock import patch
from ir_query_engine.ranker.ranking import Matcher, MatchFeatures
from ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model import TfIdfModelStruct
from ir_query_engine.retrieve_match_models.lda_feature.lda_model import LdaModelStruct
from ir_query_engine.rank_match_models.topic_word_lookup_feature.topic_word_lookup_model import TopicWordLookupModelStruct
from ir_query_engine.rank_match_models.word2vec_feature.word2vec_model import Word2VecModel


__author__ = 'Deyang'


class MatcherTestCase(unittest.TestCase):

    @patch.object(TfIdfModelStruct, 'get_cooccur_features')
    @patch.object(Word2VecModel, 'get_similarities')
    @patch.object(TopicWordLookupModelStruct, 'get_similarities')
    @patch.object(LdaModelStruct, 'get_similarities')
    @patch.object(TfIdfModelStruct, 'get_similarities')
    def test_match(self,
                   mock_tfidf_get_sims,
                   mock_lda_get_sims,
                   mock_topic_get_sims,
                   mock_word2vec_get_sims,
                   mock_get_cooccur_features):
        mock_tfidf_get_sims.side_effect = [[(0, 1.0), (1, 2.0)]]
        mock_lda_get_sims.side_effect = [[(0, 5.0), (1, 6.0)],
                                         [(0, 7.0), (1, 8.0)]]
        mock_topic_get_sims.side_effect = [[(0, 9.0), (1, 10.0)],
                                           [(0, 11.0), (1, 12.0)]]
        mock_word2vec_get_sims.side_effect = [[(0, -0.1), (1, -0.2)],
                                              [(0, -0.3), (1, -0.4)]]

        mock_get_cooccur_features.side_effect = [
            (1, 0.5, 2.0, 2.0),
            (2, 0.6, 4.0, 2.0),
            (3, 0.7, 6.0, 2.0),
            (4, 0.8, 4.0, 1.0)
        ]

        query_doc = "q"
        question_answer_pairs = [("q1", "a1"), ("q", "aq2")]

        matcher = Matcher(TfIdfModelStruct(),
                          LdaModelStruct(),
                          TopicWordLookupModelStruct(None, None),
                          Word2VecModel())
        results = matcher.match(query_doc, question_answer_pairs)
        self.assertEqual(len(results), 2)

        expected_match_feature1 = MatchFeatures(
            question_tfidf_sim=1.0,
            question_lda_sim=5.0,
            answer_lda_sim=7.0,
            question_topic_word_sim=9.0,
            answer_topic_word_sim=11.0,
            question_word2vec_sim=-0.1,
            answer_word2vec_sim=-0.3,
            answer_lcs_len=0,
            question_cooccur_size=1,
            answer_cooccur_size=3,
            question_cooccur_rate=0.5,
            answer_cooccur_rate=0.7,
            question_cooccur_sum_idf=2.0,
            answer_cooccur_sum_idf=6.0,
            question_cooccur_avg_idf=2.0,
            answer_cooccur_avg_idf=2.0,
            question_edit_distance=1,
        )

        expected_match_feature2 = MatchFeatures(
            question_tfidf_sim=2.0,
            question_lda_sim=6.0,
            answer_lda_sim=8.0,
            question_topic_word_sim=10.0,
            answer_topic_word_sim=12.0,
            question_word2vec_sim=-0.2,
            answer_word2vec_sim=-0.4,
            answer_lcs_len=1,
            question_cooccur_size=2,
            answer_cooccur_size=4,
            question_cooccur_rate=0.6,
            answer_cooccur_rate=0.8,
            question_cooccur_sum_idf=4.0,
            answer_cooccur_sum_idf=4.0,
            question_cooccur_avg_idf=2.0,
            answer_cooccur_avg_idf=1.0,
            question_edit_distance=0,
        )

        self.assertEqual(results[0].to_vec(), expected_match_feature1.to_vec())
        self.assertEqual(results[1].to_vec(), expected_match_feature2.to_vec())


if __name__ == '__main__':
    unittest.main()
