import unittest
from ir_query_engine.common import DataStore

__author__ = 'Deyang'


class DataStoreTestCase(unittest.TestCase):

    def setUp(self):
        self.data = [
          {
            "question": "tell me about a16z",
            "question_topic_words": ["a16z"],
            "answer": "a16z is a Silicon Valley-based venture capital firm with $2.7 billion under management. They invest from seed to growth.",
            "answer_topic_words": ["a16z", "venture", "capital"],
            "qa_pairs_with_matching_score": [
                {
                    "question": "tell me about a16z",
                    "answer": "a16z is a Silicon Valley-based venture capital firm with $2.7 billion under management. They invest from seed to growth.",
                    "score": 100
                },
                {
                  "question": "who is the founder of a16z",
                  "answer": "Marc Andreessen and Ben Horowitz co-founded a16z.",
                  "score": 50
                },
                {
                  "question": "Where id a16z?",
                  "answer": "a16z is located in Menlo Park",
                  "score": 10
                },
                {
                  "question": "Where is the headquarter of 500px?",
                  "answer": "500px headquarter in san francisco",
                  "score": 0
                }
              ]
          },
          {
            "question": "who is the founder of a16z",
            "question_topic_words": ["founder", "a16z"],
            "answer": "Marc Andreessen and Ben Horowitz co-founded a16z.",
            "answer_topic_words": ["co-founded", "a16z"],
            "qa_pairs_with_matching_score": [
                {
                  "question": "who is the founder of a16z",
                  "answer": "Marc Andreessen and Ben Horowitz co-founded a16z.",
                  "score": 100
                },
                {
                  "question": "who is Steven Sinofsky",
                  "answer": "Steven Sinofsky is a board partner at a16z.",
                  "score": 40
                },
                {
                  "question": "Where id a16z?",
                  "answer": "a16z is located in Menlo Park",
                  "score": 5
                }
              ]
          }
        ]

    def test_add_doc(self):
        d = DataStore({})
        doc1 = "a"
        did, added = d._add_doc(doc1)
        self.assertEqual(did, 0)
        self.assertTrue(added)
        self.assertEqual(d.doc_set, [doc1])
        self.assertEqual(d.doc_to_id, {doc1: 0})

        did, added = d._add_doc(doc1)
        self.assertEqual(did, 0)
        self.assertFalse(added)

        doc2 = "b"
        did, added = d._add_doc(doc2)
        self.assertEqual(did, 1)
        self.assertTrue(added)
        self.assertEqual(d.doc_set, [doc1, doc2])
        self.assertEqual(d.doc_to_id, {doc1: 0, doc2: 1})

    def test_init(self):
        d = DataStore(self.data)

        expected_doc_set = [
          "tell me about a16z",
          "a16z is a Silicon Valley-based venture capital firm with $2.7 billion under management. They invest from seed to growth.",
          "who is the founder of a16z",
          "Marc Andreessen and Ben Horowitz co-founded a16z.",
          "Where id a16z?",
          "a16z is located in Menlo Park",
          "Where is the headquarter of 500px?",
          "500px headquarter in san francisco",
          "who is Steven Sinofsky",
          "Steven Sinofsky is a board partner at a16z."
        ]
        self.assertEqual(d.doc_set, expected_doc_set)

        expected_doc_to_id = dict()
        for idx, doc in enumerate(expected_doc_set):
            expected_doc_to_id[doc] = idx

        self.assertEqual(d.doc_to_id, expected_doc_to_id)

        expected_question_set = [0, 2]
        self.assertEqual(d.question_set, expected_question_set)

        expected_answer_set = [1, 3]
        self.assertEqual(expected_answer_set, d.answer_set)

        expected_qid_to_qa_pair = {0: (0, 1), 2: (2, 3)}
        self.assertEqual(d.qid_to_qa_pair, expected_qid_to_qa_pair)

        expected_aid_to_qa_pairs = {1: [(0, 1)], 3: [(2, 3)]}
        self.assertEqual(d.aid_to_qa_pairs, expected_aid_to_qa_pairs)

        expected_topic_word_docs = [
          ("tell me about a16z", ["a16z"]),
          ("a16z is a Silicon Valley-based venture capital firm with $2.7 billion under management. They invest from seed to growth.",
           ["a16z", "venture", "capital"]),
          ("who is the founder of a16z", ["founder", "a16z"]),
          ("Marc Andreessen and Ben Horowitz co-founded a16z.", ["co-founded", "a16z"])
        ]
        self.assertEqual(d.topic_word_docs, expected_topic_word_docs)

        expected_rank_data =  {
            "tell me about a16z": [
                (
                    ("tell me about a16z",
                     "a16z is a Silicon Valley-based venture capital firm with $2.7 billion under management. They invest from seed to growth."),
                    100
                ),
                (
                    ("who is the founder of a16z",
                     "Marc Andreessen and Ben Horowitz co-founded a16z."),
                    50
                ),
                (
                    ("Where id a16z?",
                     "a16z is located in Menlo Park"),
                    10
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
                    ("who is Steven Sinofsky",
                     "Steven Sinofsky is a board partner at a16z."),
                    40
                ),
                (
                    ("Where id a16z?",
                     "a16z is located in Menlo Park"),
                    5
                )
            ]
        }
        self.assertEqual(d.rank_data, expected_rank_data)

    def test_getters(self):
        d = DataStore(self.data)
        self.assertEqual(d.get_all_questions(),
                         ["tell me about a16z", "who is the founder of a16z"])
        self.assertEqual(d.get_all_answers(),
                         ["a16z is a Silicon Valley-based venture capital firm with $2.7 billion under management. They invest from seed to growth.",
                          "Marc Andreessen and Ben Horowitz co-founded a16z."])
        self.assertEqual(d.get_doc_id_from_question_pos(0), 0)
        self.assertEqual(d.get_doc_id_from_answer_pos(0), 1)


if __name__ == '__main__':
    unittest.main()
