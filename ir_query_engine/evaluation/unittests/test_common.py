import unittest
from ir_query_engine.evaluation.common import split_raw_data_k_fold

__author__ = 'Deyang'


class CommonTestCase(unittest.TestCase):

    def setUp(self):
        self.data = [
          {
            "_question": "tell me about a16z",
            "_question_topic_words": ["a16z"],
            "answer": "a16z is a Silicon Valley-based venture capital firm with $2.7 billion under management. They invest from seed to growth.",
            "answer_topic_words": ["a16z", "venture", "capital"],
            "context": [{
                "company": "a16z"
            }],
            "qa_pairs_with_matching_score": [
                {
                    "_question": "tell me about a16z",
                    "answer": "a16z is a Silicon Valley-based venture capital firm with $2.7 billion under management. They invest from seed to growth.",
                    "score": 100
                },
                {
                  "_question": "who is the founder of a16z",
                  "answer": "Marc Andreessen and Ben Horowitz co-founded a16z.",
                  "score": 50
                },
                {
                  "_question": "Where id a16z?",
                  "answer": "a16z is located in Menlo Park",
                  "score": 10
                },
                {
                  "_question": "Where is the headquarter of 500px?",
                  "answer": "500px headquarter in san francisco",
                  "score": 0
                }
              ]
          },
          {
            "_question": "who is the founder of a16z",
            "_question_topic_words": ["founder", "a16z"],
            "context": [{
                "Investor": "Alex Rampell"
            }],
            "answer": "Marc Andreessen and Ben Horowitz co-founded a16z.",
            "answer_topic_words": ["co-founded", "a16z"],
            "qa_pairs_with_matching_score": [
                {
                  "_question": "who is the founder of a16z",
                  "answer": "Marc Andreessen and Ben Horowitz co-founded a16z.",
                  "score": 100
                },
                {
                  "_question": "who is Steven Sinofsky",
                  "answer": "Steven Sinofsky is a board partner at a16z.",
                  "score": 40
                },
                {
                  "_question": "Where id a16z?",
                  "answer": "a16z is located in Menlo Park",
                  "score": 5
                }
              ]
          }
        ]

    def test_split_raw_data_k_fold(self):
        train_test_data_tuples = split_raw_data_k_fold(
            self.data,
            2
        )
        # assert the first train data store
        self.assertEqual(train_test_data_tuples[0][0].doc_set,
                         ['who is the founder of a16z', 'Marc Andreessen and Ben Horowitz co-founded a16z.', 'who is Steven Sinofsky', 'Steven Sinofsky is a board partner at a16z.', 'Where id a16z?', 'a16z is located in Menlo Park'])
        self.assertEqual(train_test_data_tuples[0][0].question_set,
                         [0])
        self.assertEqual(train_test_data_tuples[0][0].answer_set,
                         [1])
        # assert the first test data set
        self.assertEqual(train_test_data_tuples[0][2].questions,
                         ['tell me about a16z'])
        self.assertEqual(train_test_data_tuples[0][2].top_answers,
                         ['a16z is a Silicon Valley-based venture capital firm with $2.7 billion under management. They invest from seed to growth.'])

        # assert the second train data store
        self.assertEqual(train_test_data_tuples[1][0].doc_set,
                         ['tell me about a16z', 'a16z is a Silicon Valley-based venture capital firm with $2.7 billion under management. They invest from seed to growth.', 'who is the founder of a16z', 'Marc Andreessen and Ben Horowitz co-founded a16z.', 'Where id a16z?', 'a16z is located in Menlo Park', 'Where is the headquarter of 500px?', '500px headquarter in san francisco'])
        self.assertEqual(train_test_data_tuples[1][0].question_set,
                         [0])
        self.assertEqual(train_test_data_tuples[1][0].answer_set,
                         [1])
        # assert the second test data set
        self.assertEqual(train_test_data_tuples[1][2].questions,
                         ['who is the founder of a16z'])
        self.assertEqual(train_test_data_tuples[1][2].top_answers,
                         ['Marc Andreessen and Ben Horowitz co-founded a16z.'])

if __name__ == '__main__':
    unittest.main()
