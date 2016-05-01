import unittest
from mock import patch
from ir_query_engine.ranker.ranking import Matcher, MatchFeatures, RankTrainingDataGenerator
import os

__author__ = 'Deyang'


class RankTrainingDataGeneratorTestCase(unittest.TestCase):

    def setUp(self):
        self.rank_data = {
            "tell me about a16z": [
                (
                    ("tell me about a16z",
                     "a16z is a Silicon Valley-based venture capital firm with $2.7 billion under management. They invest from seed to growth."),
                    100
                ),
                (
                    ("Where is the headquarter of 500px?",
                     "500px headquarter in san francisco"),
                    0
                )
            ],
            "who is the founder of a16z": [
                (
                    ("who is the founder of a16z",
                     "Marc Andreessen and Ben Horowitz co-founded a16z."),
                    100
                ),
                (
                    ("Where id a16z?",
                     "a16z is located in Menlo Park"),
                    5
                )
            ]
        }
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.test_train_data_path = os.path.join(dir_path,
                                                 'test_train.dat')

    @patch('ir_query_engine.ranker.ranking.get_train_data_path')
    @patch.object(Matcher, 'match')
    def test_write_training_data(self, mock_match, mock_train_data_path):
        mock_match.side_effect = [
            [
                MatchFeatures(
                    question_tfidf_sim=1.0,
                    # answer_tf_idf_sim=3.0,
                    question_lda_sim=5.0,
                    answer_lda_sim=7.0,
                    question_topic_word_sim=9.0,
                    # answer_topic_word_sim=11.0,
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
                ),
                MatchFeatures(
                    question_tfidf_sim=2.0,
                    # answer_tf_idf_sim=4.0,
                    question_lda_sim=6.0,
                    answer_lda_sim=8.0,
                    question_topic_word_sim=10.0,
                    # answer_topic_word_sim=12.0,
                    question_word2vec_sim=-0.2,
                    answer_word2vec_sim=-0.4,
                    answer_lcs_len=0,
                    question_cooccur_size=2,
                    answer_cooccur_size=4,
                    question_cooccur_rate=0.6,
                    answer_cooccur_rate=0.8,
                    question_cooccur_sum_idf=4.0,
                    answer_cooccur_sum_idf=4.0,
                    question_cooccur_avg_idf=2.0,
                    answer_cooccur_avg_idf=1.0,
                    question_edit_distance=1,
                )
            ],
            [
                MatchFeatures(
                    question_tfidf_sim=1.03,
                    # answer_tf_idf_sim=3.0,
                    question_lda_sim=5.0,
                    answer_lda_sim=7.0,
                    question_topic_word_sim=9.5,
                    # answer_topic_word_sim=11.0,
                    question_word2vec_sim=-0.1,
                    answer_word2vec_sim=-4.5,
                    answer_lcs_len=0,
                    question_cooccur_size=1,
                    answer_cooccur_size=3,
                    question_cooccur_rate=3.6,
                    answer_cooccur_rate=0.7,
                    question_cooccur_sum_idf=11.3,
                    answer_cooccur_sum_idf=6.0,
                    question_cooccur_avg_idf=2.0,
                    answer_cooccur_avg_idf=2.0,
                    question_edit_distance=1,
                ),
                MatchFeatures(
                    question_tfidf_sim=2.03,
                    # answer_tf_idf_sim=4.0,
                    question_lda_sim=6.0,
                    answer_lda_sim=8.0,
                    question_topic_word_sim=10.0,
                    # answer_topic_word_sim=12.0,
                    question_word2vec_sim=-0.2,
                    answer_word2vec_sim=-0.4,
                    answer_lcs_len=2,
                    question_cooccur_size=4.9,
                    answer_cooccur_size=4,
                    question_cooccur_rate=0.6,
                    answer_cooccur_rate=8.1,
                    question_cooccur_sum_idf=4.0,
                    answer_cooccur_sum_idf=4.0,
                    question_cooccur_avg_idf=2.0,
                    answer_cooccur_avg_idf=1.0,
                    question_edit_distance=1,
                )
            ]
        ]
        mock_train_data_path.return_value = self.test_train_data_path

        matcher = Matcher(None,
                          None,
                          None,
                          None)
        rank_data = list(self.rank_data.iteritems())
        rank_model = RankTrainingDataGenerator(matcher=matcher, rank_data=rank_data)

        rank_model.write_training_data()


if __name__ == '__main__':
    unittest.main()