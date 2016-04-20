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
            "ranked_answers": [
              {
              "answer": "a16z is a Silicon Valley-based venture capital firm with $2.7 billion under management. They invest from seed to growth.",
              "rank": 1
              },
              {
              "answer": "Marc Andreessen and Ben Horowitz co-founded a16z.",
              "rank": 10
              },
              {
              "answer": "a16z is located in Menlo Park",
              "rank": 10
              },
              {
              "answer": "500px headquarter in san francisco",
              "rank": 30
              }
              ]
          },
          {
            "question": "who is the founder of a16z",
            "question_topic_words": ["founder", "a16z"],
            "answer": "Marc Andreessen and Ben Horowitz co-founded a16z.",
            "answer_topic_words": ["co-founded", "a16z"],
            "ranked_answers": [
              {
              "answer": "Marc Andreessen and Ben Horowitz co-founded a16z.",
              "rank": 1
              },
              {
              "answer": "Steven Sinofsky is a board partner at a16z.",
              "rank": 20
              },
              {
              "answer": "a16z is located in Menlo Park",
              "rank": 20
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
          "a16z is located in Menlo Park",
          "500px headquarter in san francisco",
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


if __name__ == '__main__':
    unittest.main()
